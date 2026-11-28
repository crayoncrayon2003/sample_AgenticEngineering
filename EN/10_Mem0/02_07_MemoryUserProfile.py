import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, after_agent, dynamic_prompt
from langgraph.checkpoint.memory import InMemorySaver
# Disable mem0 telemetry and place its configuration folder (default ~/.mem0) under this folder's outputs.
# Set both before importing mem0.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("MEM0_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "mem0"))
from mem0 import Memory
from qdrant_client import QdrantClient

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic

# Embeddings are only available from OpenAI in this example, so Claude cannot run it.
if PROVIDER != "openai":
    sys.exit('This example uses embeddings (OpenAI only). Run with LLM_PROVIDER=openai.')


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
# Long-term memory storage (mem0)
#####################################
# mem0 handles fact extraction, embedding storage, and semantic search together.
# llm: Fact-extraction LLM. Accepts a LangChain model directly; use slightly deeper reasoning because extraction requires difficult judgments.
# embedder: How text is converted to embeddings
# vector_store: Vector storage. Use Qdrant in memory only while the program runs, as with InMemoryStore.
# history_db_path: Storage for memory change history (additions, updates, deletions)
# custom_instructions: Fact extraction instructions
os.makedirs(os.environ["MEM0_DIR"], exist_ok=True)
memory = Memory.from_config({
    "llm": {"provider": "langchain", "config": {"model": make_llm("low")}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {"provider": "qdrant", "config": {"client": QdrantClient(location=":memory:"), "embedding_model_dims": 1536}},
    "history_db_path": os.path.join(os.environ["MEM0_DIR"], "history.db"),
    "custom_instructions": (
        'Extract only facts about the user that will help future conversations, such as their name, preferences, and plans. '
        'Write each fact as one English sentence in the form "The user ...". Extract nothing from statements containing only questions or requests.'
    ),
})


#####################################
# Runtime information: Who is speaking
#####################################
@dataclass
class UserContext:
    user_id: str




#####################################
# Middleware 1: Recall
#####################################
@dynamic_prompt
def add_memories_to_prompt(request: ModelRequest) -> str:
    question = request.messages[-1].text
    # Use mem0 to search only this user's memories in order of similarity to the question (separate users via filters.user_id).
    items = memory.search(question, filters={"user_id": request.runtime.context.user_id}, top_k=3)["results"]
    memories = "\n".join(f"- {item['memory']}" for item in items) or '(none)'
    print(f"  [Recalled memories] {[item['memory'] for item in items]}")
    return (
        f"You are this user's personal assistant. Answer concisely in English. The following memories are all you know about the user. If something is not in memory, say you do not know.\n[Memories]\n{memories}"
    )


#####################################
# Middleware 2: Remember
#####################################
# 01_LangChain/02_07_MemoryUserProfile.py implements its own extraction chain and saves facts that do not duplicate existing memories.
# mem0.add extracts facts, compares them with existing memories, and automatically performs ADD, UPDATE, or DELETE operations.
@after_agent
def save_new_facts(state: AgentState, runtime) -> None:
    user_message = [m for m in state["messages"] if m.type == "human"][-1].text
    result_add = memory.add([{"role": "user", "content": user_message}], user_id=runtime.context.user_id)
    for item in result_add["results"]:
        print(f"  [Remembered ({item.get('event', 'ADD')})] {item['memory']}")
    return None


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[],
    # middleware: Recall and memory processing before and after responses
    middleware=[add_memories_to_prompt, save_new_facts],
    # context_schema: Schema for runtime information identifying the user
    context_schema=UserContext,
    # checkpointer: Conversation-specific history storage (as in 02_05)
    checkpointer=InMemorySaver(),
)


def main():
    """Example: An AI assistant remembers user names, preferences, and plans across conversations, using mem0 for storage, search, and fact extraction. The assistant separates Tanaka's (tanaka) and Suzuki's (suzuki) memories. Tanaka shares personal details, explanation preferences, preferred code language, and travel plans in thread-1, then asks about preferences/plans in thread-2. Suzuki, who shared no such details, asks about travel plans in thread-3.
    Recall (add_memories_to_prompt): Before LLM calls, memory.search retrieves the three closest memories and adds them to instructions; filters.user_id prevents cross-user retrieval. Remember (save_new_facts): After answering, pass the user's statement to memory.add. mem0 extracts facts using custom_instructions, compares existing memories, and adds, updates, or deletes them.
    Flow: Recall, answer with memories, then extract/store new facts. After all five messages, search Tanaka's memories for explanation preferences and display closest first. Unlike 01_LangChain/02_07_MemoryUserProfile.py's InMemoryStore and custom extraction/deduplication chain, mem0 handles storage, search, extraction, and comparison together. Roles, statements, and agent are unchanged. Run with LLM_PROVIDER=openai because this example's embeddings require OpenAI.
    """

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
        # thread_id: ID identifying which conversation to continue
        config = {"configurable": {"thread_id": thread_id}}

        # Run the agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
            context=UserContext(user_id=user_id),
        )
        print(f"[{user_id} / {thread_id}] AI：{result['messages'][-1].text}")

    # Search Tanaka's memories directly by meaning (closest to 'explanation preferences' first).
    print("--- Search Tanaka's memories for explanation preferences ---")
    for item in memory.search('Explanation preferences', filters={"user_id": "tanaka"}, top_k=5)["results"]:
        print(f"  Similarity {item['score']:.2f}: {item['memory']}")


if __name__ == "__main__":
    main()
