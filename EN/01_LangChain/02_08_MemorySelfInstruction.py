import json
import os

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, after_agent, dynamic_prompt
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

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
# Instruction storage
#####################################
# Save the initial instructions.
NAMESPACE = ("agent", "explainer")
store = InMemoryStore()
store.put(NAMESPACE, "instructions", {"text": 'You explain technical terms. Explain in English.'})


# Read the saved instructions.
def load_instructions(store) -> str:
    return store.get(NAMESPACE, "instructions").value["text"]


#####################################
# Middleware 1: Use saved instructions on every model call
#####################################
@dynamic_prompt
def use_saved_instructions(request: ModelRequest) -> str:
    return load_instructions(request.runtime.store)


#####################################
# Middleware 2: After a response, revise instructions if a preference is expressed
#####################################
# Define the LLM response schema (instruction revision)
class InstructionUpdate(BaseModel):
    # request: The user's preferences for future answers.
    # changed: True if the instructions should be rewritten.
    # instructions: The entire revised instructions.
    # Order request before changed so the model states the preference before deciding.
    request: str = Field(description="The user's preference for future answers; an empty string if none")
    changed: bool = Field(description='true if request is nonempty and the instructions should change')
    instructions: str = Field(description='The entire updated instructions. Return the current instructions unchanged if no update is needed.')


prompt_revise = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        "You manage an agent's instructions (system prompt).\n"
        '- Only if the user clearly requests a change to future answers (length, format, tone, etc.), '
        'set changed to true. Set it to false for ordinary questions alone.\n'
        '- When updating, add the request to the current instructions and return the entire instructions '
        "as directions to the agent starting with 'You are ...'. "
        "Do not include conversation history or explanations such as 'in this case'.",
    ),(
        # user: A message from the application user
        "user",
        '[Current instructions]\n{current}\n\n[Conversation]\n{conversation}'
    ),
])
# Use a model with slightly deeper reasoning because detecting preferences requires difficult judgments.
chain_revise = prompt_revise | make_llm("low").with_structured_output(InstructionUpdate, method="json_schema")


@after_agent
def revise_instructions(state: AgentState, runtime) -> None:
    current = load_instructions(runtime.store)
    conversation = [{"role": m.type, "content": m.text} for m in state["messages"]]
    result_revise = chain_revise.invoke({
        "current": current,
        "conversation": json.dumps(conversation, ensure_ascii=False),
    })
    if result_revise.changed:
        runtime.store.put(NAMESPACE, "instructions", {"text": result_revise.instructions})
        print(f'  [Instructions updated] {result_revise.instructions}')
    else:
        print('  [Instructions unchanged]')
    return None


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[],
    # middleware: Use and revise instructions before and after responses
    # Do not pass system_prompt; middleware 1 uses the saved instructions every time.
    middleware=[use_saved_instructions, revise_instructions],
    # store: Instruction storage
    store=store,
    # checkpointer: Conversation-specific history storage (as in 02_05)
    checkpointer=InMemorySaver(),
)


def main():
    """Example: An AI assistant remembers response preferences as its own instructions and applies them across conversations. It explains technical terms and stores response instructions. Tanaka asks about terms and states preferred response styles."""

    print(f'Initial instructions: {load_instructions(store)}')

    # Three messages (conversation ID, message)
    # Conversation 1 (thread-1): Tanaka asks the assistant what an API is.
    # The assistant answers according to the initial instructions.
    # Tanaka requests explanations within three sentences, with one familiar analogy.
    # The assistant saves instructions incorporating the request.
    # Conversation 2 (thread-2): In a different conversation, Tanaka asks what a database is.
    # Following the instructions saved in conversation 1, the assistant answers within three sentences with an analogy.
    messages = [
        ("thread-1", 'What is an API?'),
        ("thread-1", 'That is too long. From now on, explain within three sentences and include one familiar analogy.'),
        ("thread-2", 'What is a database?'),  # The "thread-id" is different.
    ]
    for thread_id, message in messages:
        print("----------------------------------------")
        print(f'[{thread_id}] User: {message}')
        # thread_id: ID identifying which conversation to continue
        config = {"configurable": {"thread_id": thread_id}}
        # Run the agent
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
        print(f"[{thread_id}] AI：{result['messages'][-1].text}")

    print(f'Final instructions: {load_instructions(store)}')


if __name__ == "__main__":
    main()
