import os
import sqlite3
import sys

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
SQLITE_PATH = os.path.join(ROOT, "outputs", "threads.sqlite3")
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "memory"  # memory・sqlite・postgres（引数で指定）


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
if BACKEND == "memory":
    # プログラムの実行中だけ保存する（終了すると消える）
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()

elif BACKEND == "sqlite":
    # outputs/threads.sqlite3 に保存する（2回実行すると、前回の会話を覚えている）
    from langgraph.checkpoint.sqlite import SqliteSaver

    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(SQLITE_PATH, check_same_thread=False))

elif BACKEND == "postgres":
    # .env の SESSION_DATABASE_URL に保存する（事前に docker compose up -d を実行する）
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit(".env に SESSION_DATABASE_URL を設定してください。")
    checkpointer = PostgresSaver(psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row))
    checkpointer.setup()  # 初回だけテーブルを作る

else:
    sys.exit("引数は memory・sqlite・postgres のいずれかを指定してください。")


#####################################
# タスク：LLMを呼ぶ
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = "あなたは親切なアシスタントです。日本語で1〜2文で答えてください。"


@task
def call_model(messages: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=system_prompt)] + messages)


#####################################
# ワークフロー（会話履歴を、自分で受け取って保存する）
#####################################
# checkpointer を渡すと、thread_id ごとに「前回の保存値（previous）」が保存される
@entrypoint(checkpointer=checkpointer)
def chat(message: str, *, previous: list | None = None) -> entrypoint.final[dict, list]:
    # previous：同じ thread_id で、前回保存した会話履歴（初回は None）
    history = previous or []
    history = add_messages(history, [HumanMessage(content=message)])

    response = call_model(history).result()
    history = add_messages(history, [response])

    # value：今回の戻り値、save：次回の previous として保存する値
    return entrypoint.final(value={"answer": response.text, "count": len(history)}, save=history)


def main():
    """
    題材：好きな食べ物を伝えた会話と、別の会話で、エージェントの覚え方を比べる。
         会話A（thread-A）で2回、会話B（thread-B）で1回質問する。
    """

    print(f"保存方法：{BACKEND}")

    # thread_id：どの会話の続きかを表すID。会話ごとに別のIDを使う
    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    # ##################################################################
    # 会話A：1回目に好きな食べ物を伝えて、2回目に好きな食べ物を確認する
    # ##################################################################
    # 1回目：好きな食べ物を伝える
    message = "私の好きな食べ物はカレーです。"
    print(f"[thread-A] 利用者：{message}")

    # ワークフローを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡す。
    result = chat.invoke(message, config_a)
    print(f"[thread-A] AI：{result['answer']}（保存済みメッセージ {result['count']}件）")


    # 2回目：同じ会話Aの続きで質問する
    message = "私の好きな食べ物は何でしたか？"
    print(f"[thread-A] 利用者：{message}")

    # ワークフローを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の会話を覚えている。
    result = chat.invoke(message, config_a)
    print(f"[thread-A] AI：{result['answer']}（保存済みメッセージ {result['count']}件）")

    # ##################################################################
    # 会話B：会話A とは異なる別の会話で、好きな食べ物を確認する
    # ##################################################################
    message = "私の好きな食べ物は何でしたか？"
    print(f"[thread-B] 利用者：{message}")

    # ワークフローを実行する。このとき、config_bで、会話Bを示すthread_id = thread-B を渡す。
    result = chat.invoke(message, config_b)
    print(f"[thread-B] AI：{result['answer']}（保存済みメッセージ {result['count']}件）")


    # 会話の保存先は、メモリ、DBなど選択できる。「会話履歴の保存先（checkpointer）」を参照

if __name__ == "__main__":
    main()
