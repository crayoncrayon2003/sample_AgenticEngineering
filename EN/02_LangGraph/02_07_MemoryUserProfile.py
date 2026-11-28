import os
import sys
from dataclasses import dataclass
from uuid import uuid4

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

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
# Long-term memory storage
#####################################
# Embed the "text" field to enable semantic search.
store = InMemoryStore(index={
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "dims": 1536,
    "fields": ["text"],
})


#####################################
# Runtime information: Who is speaking
#####################################
@dataclass
class UserContext:
    user_id: str


#####################################
# Memory extraction chain
#####################################
# Define the LLM response schema (extracted facts)
class NewFacts(BaseModel):
    # facts: Facts about the user extracted from their statements.
    facts: list[str] = Field(
        description="Information useful in future conversations, such as the user's name, preferences, and plans. "
        "One sentence per fact, in the form 'The user ...'. Return an empty list if there is only a question and no new information."
    )


prompt_extract = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'Extract only new facts about the user from their statement.\n'
        '- Extract nothing from statements containing only questions or requests (return an empty list).\n'
        '- Do not extract information already in memory.\n'
        'Examples:\n'
        '  "I am Sato. I have a cat." -> ["The user\'s name is Sato.", "The user has a cat."]\n'
        '  "Do you remember my name?" -> []',
    ),
    # user: A message from the application user
    ("user", '[Existing memories] {known}\n[User statement] {message}'),
])
# Use a model with slightly deeper reasoning because fact extraction requires difficult judgments.
chain_extract = prompt_extract | make_llm("low").with_structured_output(NewFacts, method="json_schema")


# Create a system prompt containing memories
def make_system_prompt(memories: str) -> str:
    return (
        f"You are this user's personal assistant. Answer concisely in English. The following memories are all you know about the user. If something is not in memory, say you do not know.\n[Memories]\n{memories}"
    )


#####################################
# Tasks (individual workflow steps)
#####################################
# Call the LLM
@task
def call_model(system_prompt: str, messages: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=system_prompt)] + messages)


# Extract facts worth remembering from the user's statement.
@task
def extract_facts(known: list[str], message: str) -> NewFacts:
    return chain_extract.invoke({"known": known, "message": message})


#####################################
# Agent: Recall -> answer -> remember
#####################################
# store: Long-term memory across conversations; context_schema: Runtime schema identifying the user
@entrypoint(checkpointer=InMemorySaver(), store=store, context_schema=UserContext)
def agent(message: str, *, previous: list | None = None, runtime: Runtime[UserContext]) -> dict:
    # Namespace separating memories by user
    namespace = ("users", runtime.context.user_id)
    history = add_messages(previous or [], [HumanMessage(content=message)])

    # Step 1: Recall: Search memories related to the message.
    items = runtime.store.search(namespace, query=message, limit=3)
    memories = "\n".join(f"- {item.value['text']}" for item in items) or '(none)'
    print(f"  [Recalled memories] {[item.value['text'] for item in items]}")

    # Step 2: Answer
    response = call_model(make_system_prompt(memories), history).result()
    history = add_messages(history, [response])

    # Step 3: Remember: Extract facts from the message and save them.
    known = [item.value["text"] for item in runtime.store.search(namespace, limit=20)]
    result_extract = extract_facts(known, message).result()
    for fact in result_extract.facts:
        runtime.store.put(namespace, uuid4().hex, {"text": fact})
        print(f'  [Remembered] {fact}')

    # value: Current return value; save: Conversation history stored as previous for the next call
    return entrypoint.final(value={"answer": response.text}, save=history)


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
        # thread_id: ID identifying which conversation to continue
        config = {"configurable": {"thread_id": thread_id}}

        # Run the agent
        result = agent.invoke(message, config, context=UserContext(user_id=user_id))
        answer = result["answer"]
        print(f"[{user_id} / {thread_id}] AI：{answer}")

    # Search Tanaka's memories directly by meaning (closest to 'explanation preferences' first).
    print("--- Search Tanaka's memories for explanation preferences ---")
    for item in store.search(("users", "tanaka"), query='Explanation preferences', limit=5):
        print(f"  Similarity {item.score:.2f}: {item.value['text']}")


if __name__ == "__main__":
    main()
