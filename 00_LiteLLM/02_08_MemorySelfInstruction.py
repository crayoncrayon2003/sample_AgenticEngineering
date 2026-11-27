import json
import os

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic


def make_llm(effort: str) -> dict:
    """LLM_PROVIDER に応じて、LiteLLM に渡すモデルの設定を作る。

    LiteLLM は「提供元/モデル名」の文字列で、呼び出すLLMを切り替える。
    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """LiteLLM でLLMを呼び、返ってきたメッセージを返す（LangChain の llm.invoke に当たる）。"""
    response = litellm.completion(
        model=llm["model"],
        reasoning_effort=llm["reasoning_effort"],
        max_tokens=llm["max_tokens"],
        messages=messages,
        response_format=response_format,
        tools=tools,
        timeout=60,
        num_retries=1,
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """プロンプトの {名前} に値を入れる（LangChain の ChatPromptTemplate に当たる）。"""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("minimal")

#####################################
# 指示文の保存場所
#####################################
# 最初の指示文を保存しておく
instruction_store = {"instructions": "あなたは技術用語を説明する係です。日本語で説明してください。"}


# 保存されている指示文を読み出す
def load_instructions() -> str:
    return instruction_store["instructions"]


#####################################
# 指示文を見直す処理
#####################################
# LLMのレスポンス形式を定義する（指示文の見直し結果）
class InstructionUpdate(BaseModel):
    #  request     ： 利用者の発言にある、今後の答え方への要望
    #  changed     ： 指示文を書き換えるなら True
    #  instructions： 書き換えた後の指示文全体
    #  要望（request） → 判断（changed） の順に並べ、要望を書き出してから判断させる
    request: str = Field(description="利用者の発言にある、今後の答え方への要望。無ければ空文字")
    changed: bool = Field(description="request が空でなく、指示文を変えるならtrue")
    instructions: str = Field(description="更新後の指示文全体。変えない場合は現在の指示文をそのまま返す")


prompt_revise = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたはエージェントの指示文（システムプロンプト）を管理する係です。\n"
        "- 利用者の発言に、今後の答え方への明確な要望（長さ・形式・口調など）がある場合だけ、"
        "changed を true にしてください。普通の質問だけなら false です。\n"
        "- 更新する場合は、現在の指示文に要望を加えた指示文全体を、"
        "「あなたは〜です。」で始まるエージェントへの指示として書いてください。"
        "会話の経緯や「今回のケース」などの説明は含めないでください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": "【現在の指示文】\n{current}\n\n【会話】\n{conversation}"},
]
# 要望の有無の判断は難しいため、推論を少し深くしたモデルを使う
llm_revise = make_llm("low")


# LangChain の chain_revise.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_revise(values: dict) -> InstructionUpdate:
    response = call_llm(llm_revise, format_prompt(prompt_revise, values), response_format=InstructionUpdate)
    return InstructionUpdate.model_validate_json(response.content)


#####################################
# エージェント：保存された指示文で回答する → 要望があれば指示文を書き換える
#####################################
# 会話ごとの履歴
threads = {}


def run_agent(message: str, thread_id: str) -> str:
    history = threads.get(thread_id, [])
    history.append({"role": "user", "content": message})

    # Step1 保存された指示文で回答する（LangChain では、ミドルウェア dynamic_prompt で行っていた）
    answer = call_llm(llm, [{"role": "system", "content": load_instructions()}] + history).content
    history.append({"role": "assistant", "content": answer})
    threads[thread_id] = history

    # Step2 要望があれば、指示文を書き換える（LangChain では、ミドルウェア after_agent で行っていた）
    result_revise = chain_revise({
        "current": load_instructions(),
        "conversation": json.dumps(history, ensure_ascii=False),
    })
    if result_revise.changed:
        instruction_store["instructions"] = result_revise.instructions
        print(f"  [指示文を更新] {result_revise.instructions}")
    else:
        print("  [指示文は変更なし]")
    return answer


def main():
    """
    題材：AIアシスタントが、利用者から受けた答え方の要望を自分の指示文として記憶し、別の会話でもその答え方で答える。

    登場人物：
      AIアシスタント：このプログラムのエージェント。技術用語を説明する。答え方の指示文を保存し、会話をまたいで使う
      田中さん      ：田中さんは、AIアシスタントへ、技術用語を質問し、答え方への要望を伝える
    """

    print(f"最初の指示文：{load_instructions()}")

    # 3件の発言（会話のID、発言）
    #  会話1（thread-1）：田中さんは、AIアシスタントへ「APIとは何ですか？」と質問する。
    #                    AIアシスタントは、最初の指示文に従って答える。
    #                    田中さんは、AIアシスタントへ「3文以内で、身近な例え話を1つ入れて」と答え方の要望を伝える。
    #                    AIアシスタントは、要望を書き加えた指示文を保存する。
    #  会話2（thread-2）：田中さんは、会話1とは異なる会話2で、AIアシスタントへ「データベースとは何ですか？」と質問する。
    #                    AIアシスタントは、会話1で保存した指示文に従い、3文以内で例え話を入れて答える。
    messages = [
        ("thread-1", "APIとは何ですか？"),
        ("thread-1", "長すぎます。今後は3文以内で、身近な例え話を1つ入れて説明してください。"),
        ("thread-2", "データベースとは何ですか？"),  # "thread-id"が異なる
    ]
    for thread_id, message in messages:
        print("----------------------------------------")
        print(f"[{thread_id}] 利用者：{message}")
        # エージェントを実行する
        answer = run_agent(message, thread_id)
        print(f"[{thread_id}] AI：{answer}")

    print(f"最終的な指示文：{load_instructions()}")


if __name__ == "__main__":
    main()
