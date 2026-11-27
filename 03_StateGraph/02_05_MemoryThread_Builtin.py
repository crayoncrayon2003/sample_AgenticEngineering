import os

from dotenv import load_dotenv
from langchain.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

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
# 保存先の切り替え（sqlite・postgres）は 02_05_MemoryThread.py を参照。ここでは要約の動きに絞り、メモリに保存する
checkpointer = InMemorySaver()


#####################################
# 会話履歴が長くなったら、古い部分を要約する（LangChain の SummarizationMiddleware に当たる）
#####################################
# 02_05_MemoryThread.py では、会話履歴はすべて残り、質問するたびに増え続ける。
# ここでは、履歴が MAX_MESSAGES を超えたら、古いメッセージを1つの要約に置き換え、
# 新しいメッセージだけを KEEP_MESSAGES の数だけ残す。LLMに渡す量が増え続けるのを防げる
MAX_MESSAGES = 6  # 要約を始める条件。メッセージが6件を超えたら要約する（教材用に小さくしている）
KEEP_MESSAGES = 2  # 要約せずに、そのまま残す新しいメッセージの数
SUMMARY_PREFIX = "これまでの会話の要約："

summary_prompt = (
    "あなたは会話の要約係です。次の会話から、今後の会話に必要な事実（利用者の好み、決めたことなど）を落とさず、"
    "日本語で簡潔に要約してください。"
)

# system_prompt : エージェントに設定する役割
system_prompt = "あなたは親切なアシスタントです。日本語で1〜2文で答えてください。"


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# MessagesState：会話の履歴（messages）を持つ状態。返したメッセージは履歴に追加される

# 古いメッセージを1つの要約にまとめ、履歴を「要約＋新しいメッセージ」に入れ替える
def node_summarize(state: MessagesState) -> dict:
    old, keep = state["messages"][:-KEEP_MESSAGES], state["messages"][-KEEP_MESSAGES:]
    conversation = "\n".join(f"{m.type}: {m.text}" for m in old)
    summary = llm.invoke([SystemMessage(content=summary_prompt), HumanMessage(content=conversation)]).text
    # 返したメッセージは履歴の末尾に追加されるため、いったん全件を消して（REMOVE_ALL_MESSAGES）、
    # 要約を先頭にした並びで入れ直す
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(content=f"{SUMMARY_PREFIX}\n{summary}")] + keep}


# LLMを呼ぶ
def node_model(state: MessagesState) -> dict:
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# 条件分岐：履歴が長ければ要約へ。そうでなければ、そのままLLMへ
def select_next(state: MessagesState) -> str:
    if len(state["messages"]) > MAX_MESSAGES:
        return "summarize"
    return "model"


#####################################
# グラフを組み立てる
#####################################
#  START ─┬→ summarize → model → END（履歴が6件を超える）
#         └────────────→ model → END（6件以下）
graph = StateGraph(MessagesState)
graph.add_node("summarize", node_summarize)
graph.add_node("model", node_model)
graph.add_conditional_edges(START, select_next, ["summarize", "model"])
graph.add_edge("summarize", "model")
graph.add_edge("model", END)
# checkpointer を渡すと、thread_id ごとに状態（会話履歴）が保存される
app = graph.compile(checkpointer=checkpointer)


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
        result = app.invoke({"messages": [{"role": "user", "content": message}]}, config)
        print(f"[thread-A] AI：{result['messages'][-1].text}")

        # 保存されている会話履歴の件数と、要約が含まれているかを表示する
        history = result["messages"]
        summaries = [m for m in history if m.text.startswith(SUMMARY_PREFIX)]
        print(f"  （保存済みメッセージ {len(history)}件{'、うち1件は要約' if summaries else ''}）")
        if summaries:
            body = summaries[0].text.removeprefix(SUMMARY_PREFIX).strip()
            print(f"  要約の内容（先頭150文字）：{body[:150].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
