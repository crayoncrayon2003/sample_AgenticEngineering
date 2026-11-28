import os
from typing import NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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


llm = make_llm("low")  # With minimal reasoning, the model may say "added" without calling a tool.


#####################################
# Define agent state
#####################################
class ShopState(AgentState):
    # Add custom fields to AgentState's messages (conversation history).
    # cart: Shopping cart contents
    # turn_count: Number of agent responses
    cart: NotRequired[list[str]]
    turn_count: NotRequired[int]


#####################################
# Tools: Read and update state
#####################################
@tool
def add_to_cart(item: str, runtime: ToolRuntime) -> Command:
    """Add an item to the shopping cart.

    Args:
        item: Name of the item to add.
    """
    cart = runtime.state.get("cart", []) + [item]
    # Use Command to update state fields and the tool result message together.
    return Command(update={
        "cart": cart,
        "messages": [ToolMessage(f'Added {item}. Cart: {cart}', tool_call_id=runtime.tool_call_id)],
    })


@tool
def show_cart(runtime: ToolRuntime) -> str:
    """Return the shopping cart contents."""
    cart = runtime.state.get("cart", [])
    return f'Cart contents: {cart}' if cart else 'The cart is empty.'


#####################################
# Middleware: Count each completed response
#####################################
@after_agent(state_schema=ShopState)
def count_turns(state: ShopState, runtime) -> dict:
    return {"turn_count": state.get("turn_count", 0) + 1}


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[add_to_cart, show_cart],
    # state_schema: Agent state schema, including custom cart and turn_count fields
    state_schema=ShopState,
    # middleware: Processing before and after responses
    middleware=[count_turns],
    # checkpointer: State storage, saved per thread_id
    checkpointer=InMemorySaver(),
    # system_prompt: The agent's role
    system_prompt=(
        'You manage a shopping cart.\n'
        '- Always call add_to_cart for each item the user asks to add.\n'
        '- Always call show_cart when asked about the cart contents.\n'
        '- Never say "added" or similar without calling a tool.\n'
        'Answer briefly in English based on the tool results.'
    ),
)


def main():
    """Example: An agent that manages a shopping cart, saving cart contents and the response count in state. Three requests in conversation A (thread-A) and one in B (thread-B)."""

    # thread_id identifies which conversation to continue. Use a separate ID for each conversation.
    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    # ##################################################################
    # Conversation A: Add an apple first, milk second, and check the cart on the third turn.
    # ##################################################################
    # First turn: Add an apple
    message = 'Please add an apple to the cart.'
    print(f'[thread-A] User: {message}')

    # Run the agent with config_a passing thread_id = thread-A for conversation A.
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # Retrieve conversation A's state and display custom fields (cart and turn_count).
    state = agent.get_state(config_a).values
    print(f"[thread-A] State: cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # Second turn: Add milk in the same conversation A
    message = 'Please add milk too.'
    print(f'[thread-A] User: {message}')

    # Run the agent with config_a and thread_id = thread-A to inherit conversation A's previous state (cart contents).
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # Retrieve conversation A's state and display custom fields (cart and turn_count).
    state = agent.get_state(config_a).values
    print(f"[thread-A] State: cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # Third turn: Check the cart in the same conversation A
    message = 'What is in the cart now?'
    print(f'[thread-A] User: {message}')

    # Run the agent with config_a and thread_id = thread-A to inherit conversation A's previous state (cart contents).
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # Retrieve conversation A's state and display custom fields (cart and turn_count).
    state = agent.get_state(config_a).values
    print(f"[thread-A] State: cart={state.get('cart')}, turn_count={state.get('turn_count')}")

    # ##################################################################
    # Conversation B: Check the cart in a different conversation from A.
    # ##################################################################
    message = 'Show me the cart contents.'
    print(f'[thread-B] User: {message}')

    # Run the agent with config_b passing thread_id = thread-B for conversation B.
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_b)
    print(f"[thread-B] AI：{result['messages'][-1].text}")

    # Retrieve conversation B's state and display cart and turn_count. Its cart is empty because the state is separate from A.
    state = agent.get_state(config_b).values
    print(f"[thread-B] State: cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # See 'Define agent state (ShopState)' for the schema. State is saved per conversation in the checkpointer.

if __name__ == "__main__":
    main()
