import os
import sqlite3
import sys

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
SQLITE_PATH = os.path.join(ROOT, "outputs", "threads.sqlite3")
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "memory"  # memory, sqlite, or postgres (specified as an argument)


def make_llm(effort: str):
    """Create an OpenAI or Claude chat model based on LLM_PROVIDER. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
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
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


llm = make_llm("minimal")


#####################################
# Conversation history storage (checkpointer)
#####################################
if BACKEND == "memory":
    # Store data only while the program runs (lost on exit).
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()

elif BACKEND == "sqlite":
    # Store data in outputs/threads.sqlite3 (a second run remembers the previous conversation).
    from langgraph.checkpoint.sqlite import SqliteSaver

    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(SQLITE_PATH, check_same_thread=False))

elif BACKEND == "postgres":
    # Save using SESSION_DATABASE_URL in .env (run docker compose up -d first).
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit('Set SESSION_DATABASE_URL in .env.')
    checkpointer = PostgresSaver(psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row))
    checkpointer.setup()  # Create tables only on the first setup.

else:
    sys.exit('Specify memory, sqlite, or postgres as the argument.')


#####################################
# Task: Call the LLM
#####################################
# system_prompt: The agent's role
system_prompt = 'You are a helpful assistant. Answer in English in one or two sentences.'


@task
def call_model(messages: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=system_prompt)] + messages)


#####################################
# Workflow (manually receive and save conversation history)
#####################################
# Passing checkpointer saves the previous value per thread_id.
@entrypoint(checkpointer=checkpointer)
def chat(message: str, *, previous: list | None = None) -> entrypoint.final[dict, list]:
    # previous: Previously saved conversation history for this thread_id (None on the first call)
    history = previous or []
    history = add_messages(history, [HumanMessage(content=message)])

    response = call_model(history).result()
    history = add_messages(history, [response])

    # value: Return value for this call; save: Value stored as previous for the next call
    return entrypoint.final(value={"answer": response.text, "count": len(history)}, save=history)


def main():
    """Example: Compare memory within a conversation about favorite food and across separate conversations. Ask twice in conversation A (thread-A) and once in B (thread-B)."""

    print(f'Storage backend: {BACKEND}')

    # thread_id identifies which conversation to continue. Use a separate ID for each conversation.
    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    # ##################################################################
    # Conversation A: State a favorite food first, then ask about it on the second turn.
    # ##################################################################
    # First turn: State a favorite food
    message = 'My favorite food is curry.'
    print(f'[thread-A] User: {message}')

    # Run the workflow with config_a passing thread_id = thread-A for conversation A.
    result = chat.invoke(message, config_a)
    print(f"[thread-A] AI: {result['answer']} ({result['count']} saved messages)")


    # Second turn: Ask a follow-up in the same conversation A
    message = 'What was my favorite food?'
    print(f'[thread-A] User: {message}')

    # Run the workflow with config_a and thread_id = thread-A to remember conversation A's previous messages.
    result = chat.invoke(message, config_a)
    print(f"[thread-A] AI: {result['answer']} ({result['count']} saved messages)")

    # ##################################################################
    # Conversation B: Ask about the favorite food in a different conversation from A.
    # ##################################################################
    message = 'What was my favorite food?'
    print(f'[thread-B] User: {message}')

    # Run the workflow with config_b passing thread_id = thread-B for conversation B.
    result = chat.invoke(message, config_b)
    print(f"[thread-B] AI: {result['answer']} ({result['count']} saved messages)")


    # Choose memory, a database, or another backend. See conversation history storage (checkpointer).

if __name__ == "__main__":
    main()
