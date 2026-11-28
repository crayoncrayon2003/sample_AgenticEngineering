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

#####################################
# Fictional location data
#####################################
PRECISE_LOCATIONS = {
    '1-2-3 Chuo, Sample City': 'Sample Building A (a three-story office building)',
}
GENERAL_AREAS = {
    'Sample City': 'A fictional city for teaching. Commercial areas surround the station.',
}


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
# Catch tool errors and report them to the AI (equivalent to LangChain's wrap_tool_call middleware).
#####################################
# If a tool raises an exception, return a failure result to the AI instead of stopping the program.
# The AI can use that result to switch to an alternative method (tool 2).
def run_tool(name: str, args: dict) -> str:
    print(f'  [Tool call] {name}({args})')
    try:
        return tool_functions[name](**args)
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
    """Example: Recover from tool failure by switching methods. Three requests cover a detailed address match, a city-only match after detailed lookup fails, and no match at either level."""

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
