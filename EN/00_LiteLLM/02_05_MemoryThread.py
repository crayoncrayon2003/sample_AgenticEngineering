import json
import os
import sqlite3
import sys

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
SQLITE_PATH = os.path.join(ROOT, "outputs", "threads.sqlite3")
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "memory"  # memory, sqlite, or postgres (specified as an argument)


def make_llm(effort: str) -> dict:
    """Create LiteLLM model settings based on LLM_PROVIDER. LiteLLM selects the LLM using a 'provider/model' string. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """Call the LLM through LiteLLM and return its message (equivalent to LangChain's llm.invoke)."""
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
    """Fill named placeholders in the prompt with values (equivalent to LangChain's ChatPromptTemplate)."""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("minimal")


#####################################
# Conversation history storage (equivalent to LangChain's checkpointer)
#####################################
# Each backend has just two operations: load and save.
class MemoryHistory:
    """Store data only while the program runs (lost on exit)."""

    def __init__(self):
        self.threads = {}

    def load(self, thread_id: str) -> list:
        return self.threads.get(thread_id, [])

    def save(self, thread_id: str, messages: list) -> None:
        self.threads[thread_id] = messages


class SqliteHistory:
    """Store data in outputs/threads.sqlite3 (a second run remembers the previous conversation)."""

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
    """Store data using SESSION_DATABASE_URL in .env (run docker compose up -d first)."""

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
        sys.exit('Set SESSION_DATABASE_URL in .env.')
    history_store = PostgresHistory(url)
else:
    sys.exit('Specify memory, sqlite, or postgres as the argument.')


#####################################
# Agent (manually load and save conversation history)
#####################################
# system_prompt: The agent's role
system_prompt = 'You are a helpful assistant. Answer in English in one or two sentences.'


def chat(message: str, thread_id: str) -> dict:
    # Load the previously saved conversation history using the same thread_id (empty on the first call).
    history = history_store.load(thread_id)
    history.append({"role": "user", "content": message})

    answer = call_llm(llm, [{"role": "system", "content": system_prompt}] + history).content
    history.append({"role": "assistant", "content": answer})

    # Save conversation history for the next call.
    history_store.save(thread_id, history)
    return {"answer": answer, "count": len(history)}


def main():
    """Example: Compare memory within a conversation about favorite food and across separate conversations. Ask twice in conversation A (thread-A) and once in B (thread-B)."""

    print(f'Storage backend: {BACKEND}')

    # thread_id identifies which conversation to continue. Use a separate ID for each conversation.
    thread_a = "thread-A"
    thread_b = "thread-B"

    # ##################################################################
    # Conversation A: State a favorite food first, then ask about it on the second turn.
    # ##################################################################
    # First turn: State a favorite food
    message = 'My favorite food is curry.'
    print(f'[thread-A] User: {message}')

    # Run the agent with thread_id = thread-A to identify conversation A.
    result = chat(message, thread_a)
    print(f"[thread-A] AI: {result['answer']} ({result['count']} saved messages)")


    # Second turn: Ask a follow-up in the same conversation A
    message = 'What was my favorite food?'
    print(f'[thread-A] User: {message}')

    # Run the agent with thread_id = thread-A for conversation A; it remembers the earlier conversation.
    result = chat(message, thread_a)
    print(f"[thread-A] AI: {result['answer']} ({result['count']} saved messages)")

    # ##################################################################
    # Conversation B: Ask about the favorite food in a different conversation from A.
    # ##################################################################
    message = 'What was my favorite food?'
    print(f'[thread-B] User: {message}')

    # Run the agent with thread_id = thread-B to identify conversation B.
    result = chat(message, thread_b)
    print(f"[thread-B] AI: {result['answer']} ({result['count']} saved messages)")


    # Choose memory, a database, or another storage backend. See the conversation history storage section.

if __name__ == "__main__":
    main()
