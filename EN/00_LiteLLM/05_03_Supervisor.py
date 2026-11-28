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


llm = make_llm("low")


# Ask an agent and return its final answer (LLM -> tool execution -> LLM loop shared by all three agents).
def ask(system_prompt: str, tools: list, message: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            result = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    return response.content


#####################################
# Specialist 1: Booking agent
#####################################
def book_travel(request: str) -> str:
    """Process flight or hotel bookings (simulated; no actual reservation).

    Args:
        request: Booking request.
    """
    print('      -> [Booking tool] Simulated booking processing.')
    return f"Received booking request '{request}' (simulated processing; no actual reservation)."


# system_prompt: Booking agent's role
system_prompt_booking = (
    'You handle flight and hotel bookings. Always use book_travel for requests and report the result briefly in English. '
    'Do not add details absent from the tool result (dates, options, prices, etc.).'
)
tools_booking = [
    {"type": "function", "function": {
        "name": "book_travel",
        "description": 'Process flight or hotel bookings (simulated; no actual reservation).',
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": 'Booking request'}},
            "required": ["request"],
        },
    }},
]


#####################################
# Specialist 2: Information agent
#####################################
# Information available to the information agent (mock data)
INFO = {
    'Tourist attractions in Paris': 'Eiffel Tower, Louvre Museum, Arc de Triomphe, and Montmartre.',
    "The world's highest mountain": 'Mount Everest (8,848 meters above sea level).',
}


def search_info(topic: str) -> str:
    """Search general information (mock data).

    Args:
        topic: Topic to look up (choose from the available options).
    """
    print(f"      -> [Information tool] Searched for '{topic}'.")
    return INFO.get(topic, f"No information was found for '{topic}'.")


# system_prompt: Information agent's role
system_prompt_info = (
    'You handle general questions. Always look up requests using search_info. '
    'Answer in English using only search results. If nothing is found, say so.'
)
tools_info = [
    {"type": "function", "function": {
        "name": "search_info",
        "description": 'Search general information (mock data).',
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string", "enum": ['Tourist attractions in Paris', "The world's highest mountain"], "description": 'Topic to look up (choose from the available options)'}},
            "required": ["topic"],
        },
    }},
]


#####################################
# Tools for the coordinator: Delegate work to specialists
#####################################
def delegate_booking(request: str) -> str:
    """Delegate flight or hotel booking requests to the booking agent.

    Args:
        request: Booking request.
    """
    print('  -> [Coordinator] Delegating to the booking agent.')
    return ask(system_prompt_booking, tools_booking, request)


def delegate_info(question: str) -> str:
    """Delegate general questions to the information agent.

    Args:
        question: The question.
    """
    print('  -> [Coordinator] Delegating to the information agent.')
    return ask(system_prompt_info, tools_info, question)


#####################################
# Coordinator agent
#####################################
# system_prompt: Coordinator's role
system_prompt_coordinator = (
    'You receive user requests as a coordinator. Delegate work to specialists according to the request. '
    'Do not answer directly from your own knowledge.\n'
    '- Flight or hotel bookings: delegate_booking (always delegate even if dates or other details are missing).\n'
    '- General questions: delegate_info\n'
    '- Requests containing both a booking and a question: Delegate to both specialists.\n'
    '- Requests with unclear intent: Ask the user for details without using tools.\n'
    'Combine specialist results into one concise English answer. '
    'Do not add details absent from specialist results (dates, options, prices, etc.). '
    'Do not describe simulated processing as actually completed.'
)
tools_coordinator = [
    {"type": "function", "function": {
        "name": "delegate_booking",
        "description": 'Delegate flight or hotel booking requests to the booking agent.',
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": 'Booking request'}},
            "required": ["request"],
        },
    }},
    {"type": "function", "function": {
        "name": "delegate_info",
        "description": 'Delegate general questions to the information agent.',
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": 'The question'}},
            "required": ["question"],
        },
    }},
]

# Lookup table mapping tool names to functions (shared by all three agents)
tool_functions = {"book_travel": book_travel, "search_info": search_info, "delegate_booking": delegate_booking, "delegate_info": delegate_info}


def main():
    """Example: A coordinator receives requests, delegates to specialist agents, and combines results. Requests with multiple needs go to multiple specialists.
    Roles: The coordinator selects specialists and combines their results. The booking agent processes flights and hotels. The information agent looks up general information such as attractions.
    Four requests: 1. Booking only -> booking agent. 2. Question only -> information agent. 3. Both -> both specialists, with results combined. 4. Unclear -> ask the user for clarification without delegation.
    Flow per request: The coordinator reads and routes it; selected specialists use their tools and return results; the coordinator combines the results for the user.
    """

    # Four requests
    requests = [
        'Book a hotel in Paris.',
        'Tell me about tourist attractions in Paris.',
        'Book a hotel in Paris and tell me about tourist attractions there.',
        'Please handle that matter.',
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')
        # Run the coordinator agent.
        answer = ask(system_prompt_coordinator, tools_coordinator, request)
        print(f'Final answer: {answer}')


if __name__ == "__main__":
    main()
