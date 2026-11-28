import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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
# For sqlite/postgres selection, see 02_05_MemoryThread.py.
# 02_05_MemoryThread_Builtin.py focuses on summarization and in-memory storage.
checkpointer = InMemorySaver()


#####################################
# Built-in middleware: Summarize older messages when history grows long
#####################################
# 02_05_MemoryThread.py retains the entire history, which keeps growing with each question.
# 02_05_MemoryThread_Builtin.py replaces older messages with one summary when the history exceeds trigger.
# Retain keep recent messages to prevent unlimited growth of input sent to the LLM.
summarization = SummarizationMiddleware(
    # model: LLM used for summarization
    model=llm,
    # trigger: When to summarize; here, when the history exceeds six messages
    trigger=("messages", 6),
    # keep: Number of recent messages retained without summarization
    keep=("messages", 2),
)


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[],
    # middleware: Conversation history summarization
    middleware=[summarization],
    # checkpointer: Conversation history storage, saved per thread_id
    checkpointer=checkpointer,
    # system_prompt: The agent's role
    system_prompt='You are a helpful assistant. Answer in English in one or two sentences.',
)


def main():
    """Example: Send five messages in one conversation (thread-A) and observe the history size. State a favorite food first, discuss other topics on turns 2-4, and ask about the food on turn 5. Older messages become a summary once the history exceeds six messages, preventing unlimited growth. Check whether the summary still preserves the favorite food."""

    # thread_id: ID identifying which conversation to continue
    config = {"configurable": {"thread_id": "thread-A"}}

    messages = [
        'My favorite food is curry.',
        'The weather is nice today.',
        'Tell me how to choose a good walking route.',
        'Also suggest things to do on a rainy day.',
        'What was my favorite food?',
    ]
    for message in messages:
        print("----------------------------------------")
        print(f'[thread-A] User: {message}')
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
        print(f"[thread-A] AI：{result['messages'][-1].text}")

        # Display the saved message count and whether a summary is included.
        # The summary is prepended to history as a message marked lc_source="summarization" in additional_kwargs.
        history = result["messages"]
        summaries = [m for m in history if m.additional_kwargs.get("lc_source") == "summarization"]
        print(f"  ({len(history)} saved messages{', including one summary' if summaries else ''})")
        if summaries:
            # Display without the fixed prefix 'Here is a summary of the conversation to date:'.
            body = summaries[0].text.split("\n\n", 1)[-1]
            print(f"  Summary (first 150 characters): {body[:150].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
