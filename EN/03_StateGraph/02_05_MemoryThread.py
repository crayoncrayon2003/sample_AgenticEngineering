import os
import sqlite3
import sys

from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

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
# Node: Call the LLM
#####################################
# system_prompt: The agent's role
system_prompt = 'You are a helpful assistant. Answer in English in one or two sentences.'


# MessagesState holds conversation history (messages). Returned messages are appended to history.
def node_model(state: MessagesState) -> dict:
    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


#####################################
# Build the graph
#####################################
#  START → model → END
graph = StateGraph(MessagesState)
graph.add_node("model", node_model)
graph.add_edge(START, "model")
graph.add_edge("model", END)
# Passing checkpointer saves state (conversation history) per thread_id.
app = graph.compile(checkpointer=checkpointer)


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

    # Run the graph with config_a passing thread_id = thread-A for conversation A.
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI: {result['messages'][-1].text} ({len(result['messages'])} saved messages)")


    # Second turn: Ask a follow-up in the same conversation A
    message = 'What was my favorite food?'
    print(f'[thread-A] User: {message}')

    # Run the graph with config_a and thread_id = thread-A to remember conversation A's previous messages.
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI: {result['messages'][-1].text} ({len(result['messages'])} saved messages)")

    # ##################################################################
    # Conversation B: Ask about the favorite food in a different conversation from A.
    # ##################################################################
    message = 'What was my favorite food?'
    print(f'[thread-B] User: {message}')

    # Run the graph with config_b passing thread_id = thread-B for conversation B.
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_b)
    print(f"[thread-B] AI: {result['messages'][-1].text} ({len(result['messages'])} saved messages)")


    # Choose memory, a database, or another backend. See conversation history storage (checkpointer).

if __name__ == "__main__":
    main()
