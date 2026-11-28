import os

import litellm
from dotenv import load_dotenv

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
# エージェント1：調査担当
#####################################
# system_prompt : エージェントに設定する役割
system_prompt_researcher = (
    "あなたはAI分野の調査・整理担当です。テーマについて、主要な動向・実用例・影響・注意点を日本語で整理してください。"
    "検索ツールは無いため、あなたが持つ知識の範囲で答えてください。"
    "不確かな内容は不確かと明記し、確認していない出典やURLは作らないでください。"
)


# 調査担当に依頼し、回答を返す（ツールが無いので、LLMを1回呼ぶだけ）
def agent_researcher(message: str) -> str:
    messages = [{"role": "system", "content": system_prompt_researcher}, {"role": "user", "content": message}]
    return call_llm(llm, messages).content


#####################################
# エージェント2：執筆担当
#####################################
# system_prompt : エージェントに設定する役割
system_prompt_writer = (
    "あなたは技術記事の執筆担当です。渡された整理結果に基づいて、一般の読者向けのブログ記事を日本語で書いてください。"
    "タイトルと見出しを付け、200字程度にまとめてください。"
    "整理結果で不確かとされた内容は断定せず、根拠の無い数値や出典を追加しないでください。"
)


# 執筆担当に依頼し、回答を返す（ツールが無いので、LLMを1回呼ぶだけ）
def agent_writer(message: str) -> str:
    messages = [{"role": "system", "content": system_prompt_writer}, {"role": "user", "content": message}]
    return call_llm(llm, messages).content


def main():
    """
    題材：2人のエージェントが順番に仕事を受け渡す。調査・整理担当が整理し、その結果を執筆担当が記事にする。

    登場人物：
      調査担当：テーマについて、主要な動向を整理する
      執筆担当：調査担当から受け取った結果を、ブログ記事にする
    """

    topic = "AIエージェントの主要な動向"
    print(f"テーマ：{topic}")

    # 1段目：調査・整理担当が、テーマを整理する
    findings = agent_researcher(f"テーマ：{topic}\n主要な動向を3つ選び、それぞれ要点・実用例・影響・注意点を整理してください。")
    print("--- 1段目：調査・整理担当の整理結果 ---")
    print(findings)

    # 2段目：執筆担当が、整理結果を受け取って記事にする
    article = agent_writer(f"テーマ：{topic}\n次の整理結果に基づいて、ブログ記事を書いてください。\n\n【整理結果】\n{findings}")
    print("--- 2段目：執筆担当の記事 ---")
    print(article)


if __name__ == "__main__":
    main()
