import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import InjectedToolArg, tool
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


llm = make_llm("low")  # With minimal reasoning, the model may say "added" without calling a tool.


#####################################
# Tools: Receive and process cart contents (cart)
#####################################
# InjectedToolArg: An argument supplied by the program and hidden from the LLM. The workflow supplies cart contents.
@tool
def add_to_cart(item: str, cart: Annotated[list[str], InjectedToolArg]) -> list[str]:
    """Add an item to the shopping cart.

    Args:
        item: Name of the item to add.
    """
    return cart + [item]


@tool
def show_cart(cart: Annotated[list[str], InjectedToolArg]) -> list[str]:
    """Return the shopping cart contents."""
    return cart


llm_with_tools = llm.bind_tools([add_to_cart, show_cart])

# system_prompt: The agent's role
system_prompt = (
    'You manage a shopping cart.\n'
    '- Always call add_to_cart for each item the user asks to add.\n'
    '- Always call show_cart when asked about the cart contents.\n'
    '- Never say "added" or similar without calling a tool.\n'
    'Answer briefly in English based on the tool results.'
)


#####################################
# Task: Call the LLM
#####################################
@task
def call_model(messages: list) -> AIMessage:
    return llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)


#####################################
# Agent (manually receive and save state)
#####################################
# Passing checkpointer saves the previous value per thread_id.
@entrypoint(checkpointer=InMemorySaver())
def agent(message: str, *, previous: dict | None = None) -> dict:
    # previous: Previously saved state for the same thread_id. Start with an empty state on the first call.
    # messages: Conversation history
    # cart: Shopping cart contents
    # turn_count: Number of agent responses
    state = previous or {"messages": [], "cart": [], "turn_count": 0}
    messages = add_messages(state["messages"], [HumanMessage(content=message)])
    cart = state["cart"]

    response = call_model(messages).result()
    # While the LLM requests tool calls, execute the tools and call the LLM again.
    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "add_to_cart":
                item = tool_call["args"]["item"]
                cart = add_to_cart.invoke({"item": item, "cart": cart})
                content = f'Added {item}. Cart: {cart}'
            else:
                cart_now = show_cart.invoke({"cart": cart})
                content = f'Cart contents: {cart_now}' if cart_now else 'The cart is empty.'
            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

        messages = add_messages(messages, [response] + tool_messages)
        response = call_model(messages).result()
    messages = add_messages(messages, [response])

    # The return value becomes previous on the next call (also increment the response count here).
    return {"messages": messages, "cart": cart, "turn_count": state["turn_count"] + 1}


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
    state = agent.invoke(message, config_a)
    print(f"[thread-A] AI：{state['messages'][-1].text}")

    # Display conversation A's custom state fields (cart and turn_count).
    print(f"[thread-A] State: cart={state['cart']}, turn_count={state['turn_count']}")


    # Second turn: Add milk in the same conversation A
    message = 'Please add milk too.'
    print(f'[thread-A] User: {message}')

    # Run the agent with config_a and thread_id = thread-A to inherit conversation A's previous state (cart contents).
    state = agent.invoke(message, config_a)
    print(f"[thread-A] AI：{state['messages'][-1].text}")

    # Display conversation A's custom state fields (cart and turn_count).
    print(f"[thread-A] State: cart={state['cart']}, turn_count={state['turn_count']}")


    # Third turn: Check the cart in the same conversation A
    message = 'What is in the cart now?'
    print(f'[thread-A] User: {message}')

    # Run the agent with config_a and thread_id = thread-A to inherit conversation A's previous state (cart contents).
    state = agent.invoke(message, config_a)
    print(f"[thread-A] AI：{state['messages'][-1].text}")

    # Display conversation A's custom state fields (cart and turn_count).
    print(f"[thread-A] State: cart={state['cart']}, turn_count={state['turn_count']}")

    # ##################################################################
    # Conversation B: Check the cart in a different conversation from A.
    # ##################################################################
    message = 'Show me the cart contents.'
    print(f'[thread-B] User: {message}')

    # Run the agent with config_b passing thread_id = thread-B for conversation B.
    state = agent.invoke(message, config_b)
    print(f"[thread-B] AI：{state['messages'][-1].text}")

    # Display conversation B's cart and turn_count. Its cart is empty because its state is separate from A.
    print(f"[thread-B] State: cart={state['cart']}, turn_count={state['turn_count']}")


    # The agent's return value is saved per conversation in the checkpointer.

if __name__ == "__main__":
    main()
