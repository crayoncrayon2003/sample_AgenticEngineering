import os

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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
# For sqlite/postgres selection, see 02_05_MemoryThread.py. Here we use memory to focus on summarization.
history_store = {}


#####################################
# Summarize older messages when the history grows long (equivalent to LangChain's SummarizationMiddleware).
#####################################
# 02_05_MemoryThread.py retains the entire history, which keeps growing with each question.
# 02_05_MemoryThread_Builtin.py replaces older messages with one summary when the history exceeds MAX_MESSAGES.
# Keep only KEEP_MESSAGES recent messages to prevent the amount sent to the LLM from growing indefinitely.
MAX_MESSAGES   = 6  # Summarize when the number of messages exceeds MAX_MESSAGES.
KEEP_MESSAGES  = 2  # Number of recent messages to retain without summarizing
SUMMARY_PREFIX = 'Summary of the conversation so far:'

summary_prompt = (
    'You summarize conversations. Preserve facts needed in future turns (user preferences, decisions, etc.) from the following conversation, and '
    'summarize concisely in English.'
)


def summarize(messages: list) -> str:
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return call_llm(llm, [
        {"role": "system", "content": summary_prompt},
        {"role": "user",   "content": conversation},
    ]).content


def compress_history(history: list) -> list:
    if len(history) <= MAX_MESSAGES:
        return history
    old, keep = history[:-KEEP_MESSAGES], history[-KEEP_MESSAGES:]
    summary = summarize(old)
    # Insert the summary as a single message at the start of the history.
    return [{"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"}] + keep


#####################################
# Agent (manually load, summarize, and save conversation history)
#####################################
# system_prompt: The agent's role
system_prompt = 'You are a helpful assistant. Answer in English in one or two sentences.'


def chat(message: str, thread_id: str) -> dict:
    # Load the previously saved conversation history using the same thread_id (empty on the first call).
    history = history_store.get(thread_id, [])
    history.append({"role": "user", "content": message})

    # Summarize long history before calling the LLM.
    history = compress_history(history)

    answer = call_llm(llm, [{"role": "system", "content": system_prompt}] + history).content
    history.append({"role": "assistant", "content": answer})

    # Save conversation history for the next call.
    history_store[thread_id] = history
    return {"answer": answer, "history": history}


def main():
    """Example: Send five messages in one conversation (thread-A) and observe the history size. State a favorite food first, discuss other topics on turns 2-4, and ask about the food on turn 5. Older messages become a summary once the history exceeds six messages, preventing unlimited growth. Check whether the summary still preserves the favorite food."""

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
        result = chat(message, "thread-A")
        print(f"[thread-A] AI：{result['answer']}")

        # Display the saved message count and whether a summary is included.
        history = result["history"]
        summaries = [m for m in history if m["content"].startswith(SUMMARY_PREFIX)]
        print(f"  ({len(history)} saved messages{', including one summary' if summaries else ''})")
        if summaries:
            body = summaries[0]["content"].removeprefix(SUMMARY_PREFIX).strip()
            print(f"  Summary (first 150 characters): {body[:150].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()
