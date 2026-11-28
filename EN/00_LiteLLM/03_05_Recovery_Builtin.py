import json
import os
import time

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
            "fallbacks": ["openai/gpt-5-mini"],  # Fallback model
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "fallbacks": ["anthropic/claude-sonnet-5-5"],  # Fallback model
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
        # num_retries: Number of retries after LLM call failures (equivalent to LangChain's ModelRetryMiddleware).
        num_retries=2,
        # fallbacks: Backup models tried in order after retries fail (equivalent to LangChain's ModelFallbackMiddleware).
        fallbacks=llm["fallbacks"],
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """Fill named placeholders in the prompt with values (equivalent to LangChain's ChatPromptTemplate)."""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# Fictional location data
#####################################
PRECISE_LOCATIONS = {
    '1-2-3 Chuo, Sample City': 'Sample Building A (a three-story office building)',
}
GENERAL_AREAS = {
    'Sample City': 'A fictional city for teaching. Commercial areas surround the station.',
}

# Simulate a transient failure: Tool 2 raises a communication error on odd-numbered calls (an immediate retry succeeds).
flaky_state = {"calls": 0}


#####################################
# Tool 1: Look up detailed address information; raise an exception if not found
#####################################
def get_precise_location(address: str) -> str:
    """Look up detailed address information.

    Args:
        address: Full street address, e.g., 1-2-3 Chuo, Sample City.
    """
    if address not in PRECISE_LOCATIONS:
        raise LookupError(f'Address details not found: {address}')
    return f"{address}：{PRECISE_LOCATIONS[address]}"


#####################################
# Tool 2: Look up the general area as a fallback when the detailed address lookup fails
#####################################
def get_general_area(city: str) -> str:
    """Look up general area information as a fallback when detailed address lookup fails.

    Args:
        city: City name, e.g., Sample City.
    """
    flaky_state["calls"] += 1
    if flaky_state["calls"] % 2 == 1:
        print('  [Transient error] Communication timed out (retrying)')
        raise TimeoutError('Communication timed out.')
    if city not in GENERAL_AREAS:
        raise LookupError(f'City information also not found: {city}')
    return f"{city}：{GENERAL_AREAS[city]}"


#####################################
# Tool descriptions sent to the LLM
#####################################
# LangChain's @tool generates this description from the docstring and type hints. Here we write a dictionary manually.
tools = [
    {"type": "function", "function": {
        "name": "get_precise_location",
        "description": 'Look up detailed address information.',
        "parameters": {
            "type": "object",
            "properties": {"address": {"type": "string", "description": 'Full street address, e.g., 1-2-3 Chuo, Sample City'}},
            "required": ["address"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_general_area",
        "description": 'Look up general area information as a fallback when the detailed address lookup fails.',
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": 'City name, e.g., Sample City'}},
            "required": ["city"],
        },
    }},
]
tool_functions = {"get_precise_location": get_precise_location, "get_general_area": get_general_area}


#####################################
# Automatically retry transient tool errors (equivalent to LangChain's ToolRetryMiddleware).
#####################################
# LiteLLM handles only LLM calls. Our own code executes the tools.
# Therefore, we implement automatic tool retries ourselves.
# Retry under these conditions:
# * Retry: Transient failures (TimeoutError)
# * Do not retry: Other failures whose result stays the same on repeated attempts, such as missing data (LookupError).
TOOL_MAX_RETRIES = 2


def call_with_retry(name: str, args: dict) -> str:
    for attempt in range(TOOL_MAX_RETRIES + 1):
        try:
            return tool_functions[name](**args)
        except TimeoutError:
            if attempt == TOOL_MAX_RETRIES:
                raise
            time.sleep(0.5 * (2 ** attempt))  # Retry with increasing delays.


#####################################
# Catch tool errors and report them to the AI (equivalent to LangChain's ToolErrorMiddleware).
#####################################
# If a tool raises an exception, return a failure result to the AI instead of stopping the program.
# The AI can use that result to switch to an alternative method (tool 2).
def run_tool(name: str, args: dict) -> str:
    print(f'  [Tool call] {name}({args})')
    try:
        return call_with_retry(name, args)
    except LookupError as error:
        print(f'  [Tool error] {error}')
        return f'Failed: {error}'


#####################################
# Agent
#####################################
# system_prompt: The agent's role
system_prompt = (
    'You provide location information. Follow these steps.\n'
    '1. First look up address details with get_precise_location.\n'
    '2. Only if that fails, use get_general_area to look up the city instead of repeating the same search.\n'
    '3. If only city information is available, state that address details were not found.\n'
    '4. If both fail, say no information was found and do not guess.\n'
    'Answer concisely in English and mention that these are fictional teaching data.'
)

# Repeat LLM -> tool execution -> LLM until there are no tool calls (equivalent to LangChain's create_agent).
def run_agent(message: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            # Tool arguments are returned as a JSON string.
            args = json.loads(tool_call.function.arguments)
            result = run_tool(tool_call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    messages.append(response)
    return messages


def main():
    """Example: Three requests cover an address match, a city-only match after retrying a transient error, and no match after retrying. Unlike 03_05_Recovery.py, this example automatically retries transient city-search errors and configures LLM retries and fallback models."""

    # Three requests
    requests = [
        'Look up location information for 1-2-3 Chuo, Sample City.',
        'Look up location information for 9-9-9 Chuo, Sample City.',
        'Look up location information for 9-9-9 Chuo, Unregistered City.',
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')
        # Run the agent. Tool calls and errors are displayed during execution.
        messages = run_agent(request)
        print(f'Answer: {messages[-1].content}')


if __name__ == "__main__":
    main()
