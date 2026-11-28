import json
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


llm = make_llm("low")  # With minimal reasoning, the model may say "added" without calling a tool.


#####################################
# Agent state (saved per conversation)
#####################################
# Save a dictionary with the following fields for each thread_id.
# messages: Conversation history
# cart: Shopping cart contents
# turn_count: Number of agent responses
states = {}


def get_state(thread_id: str) -> dict:
    return states.get(thread_id, {"messages": [], "cart": [], "turn_count": 0})


#####################################
# Tools: Read and update state
#####################################
def add_to_cart(state: dict, item: str) -> str:
    """Add an item to the shopping cart."""
    state["cart"] = state["cart"] + [item]
    return f"Added {item}. Cart: {state['cart']}"


def show_cart(state: dict) -> str:
    """Return the shopping cart contents."""
    return f"Cart contents: {state['cart']}" if state["cart"] else 'The cart is empty.'


# Tool descriptions sent to the LLM (omit state, which is supplied by the program)
tools = [
    {"type": "function", "function": {
        "name": "add_to_cart",
        "description": 'Add an item to the shopping cart.',
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string", "description": 'Name of the item to add'}},
            "required": ["item"],
        },
    }},
    {"type": "function", "function": {
        "name": "show_cart",
        "description": 'Return the shopping cart contents.',
        "parameters": {"type": "object", "properties": {}},
    }},
]


#####################################
# Agent
#####################################
# system_prompt: The agent's role
system_prompt = (
    'You manage a shopping cart.\n'
    '- Always call add_to_cart for each item the user asks to add.\n'
    '- Always call show_cart when asked about the cart contents.\n'
    '- Never say "added" or similar without calling a tool.\n'
    'Answer briefly in English based on the tool results.'
)


def run_agent(message: str, thread_id: str) -> str:
    # Load the previously saved state for the same thread_id (empty on the first call).
    state = get_state(thread_id)
    state["messages"].append({"role": "user", "content": message})

    response = call_llm(llm, [{"role": "system", "content": system_prompt}] + state["messages"], tools=tools)
    # While the LLM requests tool calls, execute the tools and call the LLM again.
    while response.tool_calls:
        state["messages"].append(response.model_dump())
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "add_to_cart":
                result = add_to_cart(state, args["item"])
            else:
                result = show_cart(state)
            state["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, [{"role": "system", "content": system_prompt}] + state["messages"], tools=tools)
    state["messages"].append({"role": "assistant", "content": response.content})

    # Increment the count and save state after each response (handled by after_agent middleware in the LangChain example).
    state["turn_count"] += 1
    states[thread_id] = state
    return response.content


def main():
    """Example: An agent that manages a shopping cart, saving cart contents and the response count in state. Three requests in conversation A (thread-A) and one in B (thread-B)."""

    # thread_id identifies which conversation to continue. Use a separate ID for each conversation.
    thread_a = "thread-A"
    thread_b = "thread-B"

    # ##################################################################
    # Conversation A: Add an apple first, milk second, and check the cart on the third turn.
    # ##################################################################
    # First turn: Add an apple
    message = 'Please add an apple to the cart.'
    print(f'[thread-A] User: {message}')

    # Run the agent with thread_id = thread-A to identify conversation A.
    answer = run_agent(message, thread_a)
    print(f"[thread-A] AI：{answer}")

    # Retrieve conversation A's state and display custom fields (cart and turn_count).
    state = get_state(thread_a)
    print(f"[thread-A] State: cart={state['cart']}, turn_count={state['turn_count']}")


    # Second turn: Add milk in the same conversation A
    message = 'Please add milk too.'
    print(f'[thread-A] User: {message}')

    # Run the agent with thread_id = thread-A to inherit conversation A's previous state (cart contents).
    answer = run_agent(message, thread_a)
    print(f"[thread-A] AI：{answer}")

    # Retrieve conversation A's state and display custom fields (cart and turn_count).
    state = get_state(thread_a)
    print(f"[thread-A] State: cart={state['cart']}, turn_count={state['turn_count']}")


    # Third turn: Check the cart in the same conversation A
    message = 'What is in the cart now?'
    print(f'[thread-A] User: {message}')

    # Run the agent with thread_id = thread-A to inherit conversation A's previous state (cart contents).
    answer = run_agent(message, thread_a)
    print(f"[thread-A] AI：{answer}")

    # Retrieve conversation A's state and display custom fields (cart and turn_count).
    state = get_state(thread_a)
    print(f"[thread-A] State: cart={state['cart']}, turn_count={state['turn_count']}")

    # ##################################################################
    # Conversation B: Check the cart in a different conversation from A.
    # ##################################################################
    message = 'Show me the cart contents.'
    print(f'[thread-B] User: {message}')

    # Run the agent with thread_id = thread-B to identify conversation B.
    answer = run_agent(message, thread_b)
    print(f"[thread-B] AI：{answer}")

    # Retrieve conversation B's state and display cart and turn_count. Its cart is empty because the state is separate from A.
    state = get_state(thread_b)
    print(f"[thread-B] State: cart={state['cart']}, turn_count={state['turn_count']}")


    # See the agent state section for the schema. State is saved per conversation (thread_id).

if __name__ == "__main__":
    main()
