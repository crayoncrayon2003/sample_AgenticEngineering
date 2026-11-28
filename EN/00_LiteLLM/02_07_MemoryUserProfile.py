import math
import os
import sys

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic

# Embeddings are only available from OpenAI in this example, so Claude cannot run it.
if PROVIDER != "openai":
    sys.exit('This example uses embeddings (OpenAI only). Run with LLM_PROVIDER=openai.')


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
# Long-term memory storage (equivalent to LangChain's store)
#####################################
# Store a list of memories (text and embeddings) for each user ID.
memory_store = {}


# Convert text to embedding vectors (sequences of numbers).
def embed(text: str) -> list[float]:
    response = litellm.embedding(model="openai/text-embedding-3-small", input=[text])
    return response.data[0]["embedding"]


# Calculate how close two vectors are using cosine similarity.
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# Search the user's memories in order of semantic similarity.
def search_memories(user_id: str, query: str, limit: int) -> list[dict]:
    query_vector = embed(query)
    items = []
    for memory in memory_store.get(user_id, []):
        items.append({"text": memory["text"], "score": cosine_similarity(query_vector, memory["vector"])})
    items.sort(key=lambda item: item["score"], reverse=True)
    return items[:limit]


# Save one memory for the user.
def save_memory(user_id: str, text: str) -> None:
    memory_store.setdefault(user_id, []).append({"text": text, "vector": embed(text)})


#####################################
# Extract memories
#####################################
# Define the LLM response schema (extracted facts)
class NewFacts(BaseModel):
    # facts: Facts about the user extracted from their statements.
    facts: list[str] = Field(
        description="Information useful in future conversations, such as the user's name, preferences, and plans. "
        "One sentence per fact, in the form 'The user ...'. Return an empty list if there is only a question and no new information."
    )


prompt_extract = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'Extract only new facts about the user from their statement.\n'
        '- Extract nothing from statements containing only questions or requests (return an empty list).\n'
        '- Do not extract information already in memory.\n'
        'Examples:\n'
        '  "I am Sato. I have a cat." -> ["The user\'s name is Sato.", "The user has a cat."]\n'
        '  "Do you remember my name?" -> []'
    )},
    # user: A message from the application user
    {"role": "user", "content": '[Existing memories] {known}\n[User statement] {message}'},
]
# Use a model with slightly deeper reasoning because fact extraction requires difficult judgments.
llm_extract = make_llm("low")


# Equivalent to LangChain's chain_extract.invoke({...}). Accept prompt values as a dictionary.
def chain_extract(values: dict) -> NewFacts:
    response = call_llm(llm_extract, format_prompt(prompt_extract, values), response_format=NewFacts)
    return NewFacts.model_validate_json(response.content)


# Create a system prompt containing memories
def make_system_prompt(memories: str) -> str:
    return (
        f"You are this user's personal assistant. Answer concisely in English. The following memories are all you know about the user. If something is not in memory, say you do not know.\n[Memories]\n{memories}"
    )


#####################################
# Agent: Recall -> answer -> remember
#####################################
# History for each conversation
threads = {}


def run_agent(message: str, user_id: str, thread_id: str) -> str:
    history = threads.get(thread_id, [])
    history.append({"role": "user", "content": message})

    # Step 1: Recall memories related to the message (handled by dynamic_prompt middleware in the LangChain example).
    items = search_memories(user_id, message, limit=3)
    memories = "\n".join(f"- {item['text']}" for item in items) or '(none)'
    print(f"  [Recalled memories] {[item['text'] for item in items]}")

    # Step 2: Answer
    answer = call_llm(llm, [{"role": "system", "content": make_system_prompt(memories)}] + history).content
    history.append({"role": "assistant", "content": answer})
    threads[thread_id] = history

    # Step 3: Extract and store facts from the message (handled by after_agent middleware in the LangChain example).
    known = [memory["text"] for memory in memory_store.get(user_id, [])]
    result_extract = chain_extract({"known": known, "message": message})
    for fact in result_extract.facts:
        save_memory(user_id, fact)
        print(f'  [Remembered] {fact}')
    return answer


def main():
    """Example: An AI assistant remembers a user's name, preferences, and plans and recalls them in a different conversation. The assistant keeps Tanaka's and Suzuki's memories separate. Tanaka shares a name, explanation preferences, and travel plans; Suzuki does not."""

    # Five messages (user ID, conversation ID, message)
    # Conversation 1 (thread-1): Tanaka shares a name, explanation preferences, preferred code language, and travel plans.
    # The assistant saves these as Tanaka's memories.
    # Conversation 2 (thread-2): In a different conversation, Tanaka asks whether the assistant remembers their preferences and plans.
    # The assistant searches Tanaka's memories and answers using information from conversation 1.
    # Conversation 3 (thread-3): Suzuki asks the assistant about their own travel plans.
    # The assistant searches only Suzuki's memories. It does not have Tanaka's information in that memory, so it says it does not know.
    messages = [
        ("tanaka", "thread-1", 'I am Tanaka. I prefer short explanations that start with the conclusion. Please use Python for code examples.'),
        ("tanaka", "thread-1", 'I plan to travel to Kyoto next month.'),
        ("tanaka", "thread-2", 'Do you remember my preferred explanation style and code example language?'),  # The "thread-id" is different.
        ("tanaka", "thread-2", 'What are my travel plans?'),                                   # The "thread-id" is different.
        ("suzuki", "thread-3", 'What are my travel plans?'),
    ]
    for user_id, thread_id, message in messages:
        print("----------------------------------------")
        print(f'[{user_id} / {thread_id}] User: {message}')

        # Run the agent
        answer = run_agent(message, user_id, thread_id)
        print(f"[{user_id} / {thread_id}] AI：{answer}")

    # Search Tanaka's memories directly by meaning (closest to 'explanation preferences' first).
    print("--- Search Tanaka's memories for explanation preferences ---")
    for item in search_memories("tanaka", 'Explanation preferences', limit=5):
        print(f"  Similarity {item['score']:.2f}: {item['text']}")


if __name__ == "__main__":
    main()
