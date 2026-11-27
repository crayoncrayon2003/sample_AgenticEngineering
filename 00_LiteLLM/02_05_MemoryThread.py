import json
import os
import sqlite3
import sys

import litellm
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
SQLITE_PATH = os.path.join(ROOT, "outputs", "threads.sqlite3")
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "memory"  # memory・sqlite・postgres（引数で指定）


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
# 会話履歴の保存先（LangChain の checkpointer に当たる）
#####################################
# どの保存先も、load（読み出す）と save（保存する）の2つだけを持つ
class MemoryHistory:
    """プログラムの実行中だけ保存する（終了すると消える）。"""

    def __init__(self):
        self.threads = {}

    def load(self, thread_id: str) -> list:
        return self.threads.get(thread_id, [])

    def save(self, thread_id: str, messages: list) -> None:
        self.threads[thread_id] = messages


class SqliteHistory:
    """outputs/threads.sqlite3 に保存する（2回実行すると、前回の会話を覚えている）。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, messages TEXT)")

    def load(self, thread_id: str) -> list:
        row = self.conn.execute("SELECT messages FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
        return json.loads(row[0]) if row else []

    def save(self, thread_id: str, messages: list) -> None:
        self.conn.execute("REPLACE INTO threads VALUES (?, ?)", (thread_id, json.dumps(messages, ensure_ascii=False)))
        self.conn.commit()


class PostgresHistory:
    """.env の SESSION_DATABASE_URL に保存する（事前に docker compose up -d を実行する）。"""

    def __init__(self, url: str):
        import psycopg

        self.conn = psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, messages TEXT)")

    def load(self, thread_id: str) -> list:
        row = self.conn.execute("SELECT messages FROM threads WHERE thread_id = %s", (thread_id,)).fetchone()
        return json.loads(row[0]) if row else []

    def save(self, thread_id: str, messages: list) -> None:
        self.conn.execute(
            "INSERT INTO threads VALUES (%s, %s) ON CONFLICT (thread_id) DO UPDATE SET messages = EXCLUDED.messages",
            (thread_id, json.dumps(messages, ensure_ascii=False)),
        )


if BACKEND == "memory":
    history_store = MemoryHistory()
elif BACKEND == "sqlite":
    history_store = SqliteHistory(SQLITE_PATH)
elif BACKEND == "postgres":
    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit(".env に SESSION_DATABASE_URL を設定してください。")
    history_store = PostgresHistory(url)
else:
    sys.exit("引数は memory・sqlite・postgres のいずれかを指定してください。")


#####################################
# エージェント（会話履歴を、自分で読み出して保存する）
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = "あなたは親切なアシスタントです。日本語で1〜2文で答えてください。"


def chat(message: str, thread_id: str) -> dict:
    # 同じ thread_id で、前回保存した会話履歴を読み出す（初回は空）
    history = history_store.load(thread_id)
    history.append({"role": "user", "content": message})

    answer = call_llm(llm, [{"role": "system", "content": system_prompt}] + history).content
    history.append({"role": "assistant", "content": answer})

    # 次回のために、会話履歴を保存する
    history_store.save(thread_id, history)
    return {"answer": answer, "count": len(history)}


def main():
    """
    題材：好きな食べ物を伝えた会話と、別の会話で、エージェントの覚え方を比べる。
         会話A（thread-A）で2回、会話B（thread-B）で1回質問する。
    """

    print(f"保存方法：{BACKEND}")

    # thread_id：どの会話の続きかを表すID。会話ごとに別のIDを使う
    thread_a = "thread-A"
    thread_b = "thread-B"

    # ##################################################################
    # 会話A：1回目に好きな食べ物を伝えて、2回目に好きな食べ物を確認する
    # ##################################################################
    # 1回目：好きな食べ物を伝える
    message = "私の好きな食べ物はカレーです。"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、会話Aを示すthread_id = thread-A を渡す。
    result = chat(message, thread_a)
    print(f"[thread-A] AI：{result['answer']}（保存済みメッセージ {result['count']}件）")


    # 2回目：同じ会話Aの続きで質問する
    message = "私の好きな食べ物は何でしたか？"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、会話Aを示すthread_id = thread-A を渡すと、前の会話を覚えている。
    result = chat(message, thread_a)
    print(f"[thread-A] AI：{result['answer']}（保存済みメッセージ {result['count']}件）")

    # ##################################################################
    # 会話B：会話A とは異なる別の会話で、好きな食べ物を確認する
    # ##################################################################
    message = "私の好きな食べ物は何でしたか？"
    print(f"[thread-B] 利用者：{message}")

    # エージェントを実行する。このとき、会話Bを示すthread_id = thread-B を渡す。
    result = chat(message, thread_b)
    print(f"[thread-B] AI：{result['answer']}（保存済みメッセージ {result['count']}件）")


    # 会話の保存先は、メモリ、DBなど選択できる。「会話履歴の保存先」を参照

if __name__ == "__main__":
    main()
