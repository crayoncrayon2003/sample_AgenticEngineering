import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

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
# 会話履歴の保存先（checkpointer）
#####################################
# 保存先の切り替え（sqlite・postgres）は 02_05_MemoryThread.py を参照。
# 02_05_MemoryThread_Builtin.pyは、要約・メモリ保存に絞る
checkpointer = InMemorySaver()


#####################################
# 標準ミドルウェア：会話履歴が長くなったら、古い部分を要約する
#####################################
# 02_05_MemoryThread.py は、会話履歴はすべて残り、質問するたびに増え続ける。
# 02_05_MemoryThread_Builtin.py は、履歴が trigger を超えたら、古いメッセージを1つの要約に置き換える。
# 新しいメッセージは、keep の数だけ残す。LLMに渡す量が増え続けるのを防げぐ。
summarization = SummarizationMiddleware(
    # model   : 要約を作るLLM
    model=llm,
    # trigger : 要約を始める条件。ここでは、メッセージが6件を超えたら要約する
    trigger=("messages", 6),
    # keep    : 要約せずに、そのまま残す新しいメッセージの数
    keep=("messages", 2),
)


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[],
    # middleware    : 会話履歴を要約する処理
    middleware=[summarization],
    # checkpointer  : 会話履歴の保存先。thread_id ごとに会話履歴が保存される
    checkpointer=checkpointer,
    # system_prompt : エージェントに設定する役割
    system_prompt="あなたは親切なアシスタントです。日本語で1〜2文で答えてください。",
)


def main():
    """
    題材：1つの会話（thread-A）で5回話しかけ、会話履歴の件数の変化を見る。
         1回目に好きな食べ物を伝え、2〜4回目は別の話題、5回目に好きな食べ物を確認する。
         履歴が6件を超えると古い部分が要約に置き換わるため、件数は増え続けない。
         それでも、要約に残った内容から、1回目に伝えた好きな食べ物を答えられるかを確かめる。
    """

    # thread_id：どの会話の続きかを表すID
    config = {"configurable": {"thread_id": "thread-A"}}

    messages = [
        "私の好きな食べ物はカレーです。",
        "今日はいい天気ですね。",
        "おすすめの散歩コースの選び方を教えてください。",
        "雨の日の過ごし方も教えてください。",
        "私の好きな食べ物は何でしたか？",
    ]
    for message in messages:
        print("----------------------------------------")
        print(f"[thread-A] 利用者：{message}")
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
        print(f"[thread-A] AI：{result['messages'][-1].text}")

        # 保存されている会話履歴の件数と、要約が含まれているかを表示する
        # 要約は、additional_kwargs に lc_source="summarization" の印が付いたメッセージとして履歴の先頭に入る
        history = result["messages"]
        summaries = [m for m in history if m.additional_kwargs.get("lc_source") == "summarization"]
        print(f"  （保存済みメッセージ {len(history)}件{'、うち1件は要約' if summaries else ''}）")
        if summaries:
            # 先頭の定型文（Here is a summary of the conversation to date:）を除いて表示する
            body = summaries[0].text.split("\n\n", 1)[-1]
            print(f"  要約の内容（先頭150文字）：{body[:150].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
