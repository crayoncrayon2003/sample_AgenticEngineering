import os

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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
# For sqlite/postgres selection, see 02_05_MemoryThread.py. Here we use memory to focus on summarization.
checkpointer = InMemorySaver()


#####################################
# Summarize older messages when the history grows long (equivalent to LangChain's SummarizationMiddleware).
#####################################
# 02_05_MemoryThread.py retains all conversation history, which grows with every question.
# Here, when history exceeds MAX_MESSAGES, replace older messages with one summary and
# retain only KEEP_MESSAGES recent messages, preventing unlimited input growth.
MAX_MESSAGES = 6  # Summarize when the history exceeds six messages (a low threshold for teaching).
KEEP_MESSAGES = 2  # Number of recent messages to retain without summarizing
SUMMARY_PREFIX = 'Summary of the conversation so far:'

summary_prompt = (
    'You summarize conversations. Preserve facts needed in future turns (user preferences, decisions, etc.) from the following conversation, and '
    'summarize concisely in English.'
)

# system_prompt: The agent's role
system_prompt = 'You are a helpful assistant. Answer in English in one or two sentences.'


#####################################
# Tasks (individual workflow steps)
#####################################
# Combine old messages into one summary.
@task
def summarize(messages: list) -> str:
    conversation = "\n".join(f"{m.type}: {m.text}" for m in messages)
    return llm.invoke([SystemMessage(content=summary_prompt), HumanMessage(content=conversation)]).text


# Call the LLM
@task
def call_model(messages: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=system_prompt)] + messages)


#####################################
# Workflow (receive history, summarize if long, then call the LLM)
#####################################
# Passing checkpointer saves the previous value per thread_id.
@entrypoint(checkpointer=checkpointer)
def chat(message: str, *, previous: list | None = None) -> entrypoint.final[dict, list]:
    # previous: Previously saved conversation history for this thread_id (None on the first call)
    history = previous or []
    history = add_messages(history, [HumanMessage(content=message)])

    # Summarize long history before calling the LLM. Prepend the summary as one message.
    if len(history) > MAX_MESSAGES:
        old, keep = history[:-KEEP_MESSAGES], history[-KEEP_MESSAGES:]
        summary = summarize(old).result()
        history = [HumanMessage(content=f"{SUMMARY_PREFIX}\n{summary}")] + keep

    response = call_model(history).result()
    history = add_messages(history, [response])

    # value: Return value for this call; save: Value stored as previous for the next call
    return entrypoint.final(value={"answer": response.text, "history": history}, save=history)


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
        result = chat.invoke(message, config)
        print(f"[thread-A] AI：{result['answer']}")

        # Display the saved message count and whether a summary is included.
        history = result["history"]
        summaries = [m for m in history if m.text.startswith(SUMMARY_PREFIX)]
        print(f"  ({len(history)} saved messages{', including one summary' if summaries else ''})")
        if summaries:
            body = summaries[0].text.removeprefix(SUMMARY_PREFIX).strip()
            print(f"  Summary (first 150 characters): {body[:150].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
