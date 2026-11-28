import os

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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
# タスク（ワークフローの1ステップ）
#####################################
# 古いメッセージを、1つの要約にまとめる
@task
def summarize(messages: list) -> str:
    conversation = "\n".join(f"{m.type}: {m.text}" for m in messages)
    return llm.invoke([SystemMessage(content=summary_prompt), HumanMessage(content=conversation)]).text


# LLMを呼ぶ
@task
def call_model(messages: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=system_prompt)] + messages)


#####################################
# ワークフロー（会話履歴を受け取り、長ければ要約してから、LLMを呼ぶ）
#####################################
# checkpointer を渡すと、thread_id ごとに「前回の保存値（previous）」が保存される
@entrypoint(checkpointer=checkpointer)
def chat(message: str, *, previous: list | None = None) -> entrypoint.final[dict, list]:
    # previous：同じ thread_id で、前回保存した会話履歴（初回は None）
    history = previous or []
    history = add_messages(history, [HumanMessage(content=message)])

    # LLMを呼ぶ前に、長くなった履歴を要約する。要約は、履歴の先頭に1件のメッセージとして入れる
    if len(history) > MAX_MESSAGES:
        old, keep = history[:-KEEP_MESSAGES], history[-KEEP_MESSAGES:]
        summary = summarize(old).result()
        history = [HumanMessage(content=f"{SUMMARY_PREFIX}\n{summary}")] + keep

    response = call_model(history).result()
    history = add_messages(history, [response])

    # value：今回の戻り値、save：次回の previous として保存する値
    return entrypoint.final(value={"answer": response.text, "history": history}, save=history)


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
        result = chat.invoke(message, config)
        print(f"[thread-A] AI：{result['answer']}")

        # 保存されている会話履歴の件数と、要約が含まれているかを表示する
        history = result["history"]
        summaries = [m for m in history if m.text.startswith(SUMMARY_PREFIX)]
        print(f"  （保存済みメッセージ {len(history)}件{'、うち1件は要約' if summaries else ''}）")
        if summaries:
            body = summaries[0].text.removeprefix(SUMMARY_PREFIX).strip()
            print(f"  要約の内容（先頭150文字）：{body[:150].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
