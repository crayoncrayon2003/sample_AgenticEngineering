import json
import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, after_agent, dynamic_prompt
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic


def make_llm(effort: str):
    """LLM_PROVIDER に応じて、OpenAI または Claude のチャットモデルを作る。

    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL") or "gpt-5-nano",
            reasoning_effort=effort,
            use_responses_api=True,
            timeout=60,
            max_retries=1,
        )
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


llm = make_llm("minimal")

#####################################
# 指示文の保存場所
#####################################
# 最初の指示文を保存しておく
NAMESPACE = ("agent", "explainer")
store = InMemoryStore()
store.put(NAMESPACE, "instructions", {"text": "あなたは技術用語を説明する係です。日本語で説明してください。"})


# 保存されている指示文を読み出す
def load_instructions(store) -> str:
    return store.get(NAMESPACE, "instructions").value["text"]


#####################################
# ミドルウェア1：モデルを呼ぶたびに、保存された指示文を使う
#####################################
@dynamic_prompt
def use_saved_instructions(request: ModelRequest) -> str:
    return load_instructions(request.runtime.store)


#####################################
# ミドルウェア2：応答の後、要望があれば指示文を書き換える
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


prompt_revise = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたはエージェントの指示文（システムプロンプト）を管理する係です。\n"
        "- 利用者の発言に、今後の答え方への明確な要望（長さ・形式・口調など）がある場合だけ、"
        "changed を true にしてください。普通の質問だけなら false です。\n"
        "- 更新する場合は、現在の指示文に要望を加えた指示文全体を、"
        "「あなたは〜です。」で始まるエージェントへの指示として書いてください。"
        "会話の経緯や「今回のケース」などの説明は含めないでください。",
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "【現在の指示文】\n{current}\n\n【会話】\n{conversation}"
    ),
])
# 要望の有無の判断は難しいため、推論を少し深くしたモデルを使う
chain_revise = prompt_revise | make_llm("low").with_structured_output(InstructionUpdate, method="json_schema")


@after_agent
def revise_instructions(state: AgentState, runtime) -> None:
    current = load_instructions(runtime.store)
    conversation = [{"role": m.type, "content": m.text} for m in state["messages"]]
    result_revise = chain_revise.invoke({
        "current": current,
        "conversation": json.dumps(conversation, ensure_ascii=False),
    })
    if result_revise.changed:
        runtime.store.put(NAMESPACE, "instructions", {"text": result_revise.instructions})
        print(f"  [指示文を更新] {result_revise.instructions}")
    else:
        print("  [指示文は変更なし]")
    return None


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[],
    # middleware    : 応答の前後に実行する処理（指示文を使う・書き換える）
    #                 system_prompt は渡さず、ミドルウェア1が保存された指示文を毎回使う
    middleware=[use_saved_instructions, revise_instructions],
    # store         : 指示文の保存先
    store=store,
    # checkpointer  : 会話ごとの履歴の保存先（02_05と同じ）
    checkpointer=InMemorySaver(),
)


def main():
    """
    題材：AIアシスタントが、利用者から受けた答え方の要望を自分の指示文として記憶し、別の会話でもその答え方で答える。

    登場人物：
      AIアシスタント：このプログラムのエージェント。技術用語を説明する。答え方の指示文を保存し、会話をまたいで使う
      田中さん      ：田中さんは、AIアシスタントへ、技術用語を質問し、答え方への要望を伝える
    """

    print(f"最初の指示文：{load_instructions(store)}")

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
        # thread_id：どの会話の続きかを表すID
        config = {"configurable": {"thread_id": thread_id}}
        # エージェントを実行する
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
        print(f"[{thread_id}] AI：{result['messages'][-1].text}")

    print(f"最終的な指示文：{load_instructions(store)}")


if __name__ == "__main__":
    main()
