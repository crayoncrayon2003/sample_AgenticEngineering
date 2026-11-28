import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool

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


llm = make_llm("low")

#####################################
# Fictional location data
#####################################
# Detailed address data (assume one registered address)
PRECISE_LOCATIONS = {
    '1-2-3 Chuo, Sample City': 'Sample Building A (a three-story office building)',
}
# General area data (assume one registered address)
GENERAL_AREAS = {
    'Sample City': 'A fictional city for teaching. Commercial areas surround the station.',
}


###############################################################################
# Tool 1: Look up detailed address information; raise an exception if not found
###############################################################################
@tool
def get_precise_location(address: str) -> str:
    """Look up detailed address information.

    Args:
        address: Full street address, e.g., 1-2-3 Chuo, Sample City.
    """
    if address not in PRECISE_LOCATIONS:
        raise LookupError(f'Address details not found: {address}')
    return f"{address}：{PRECISE_LOCATIONS[address]}"


###############################################################################
# Tool 2: Look up the general area as a fallback when the detailed address lookup fails
###############################################################################
@tool
def get_general_area(city: str) -> str:
    """Look up general area information as a fallback when detailed address lookup fails.

    Args:
        city: City name, e.g., Sample City.
    """
    if city not in GENERAL_AREAS:
        raise LookupError(f'City information also not found: {city}')
    return f"{city}：{GENERAL_AREAS[city]}"


#####################################
# Middleware: Catch tool errors and report them to the AI
#####################################
# If a tool raises an exception, return a failure result to the AI instead of stopping the program.
# The AI can use that result to switch to an alternative method (tool 2).
@wrap_tool_call
def handle_tool_errors(request: ToolCallRequest, handler):
    try:
        # handler: Execute the tool body
        return handler(request)
    except LookupError as error:
        print(f'  [Tool error] {error}')
        return ToolMessage(
            content=f'Failed: {error}',
            tool_call_id=request.tool_call["id"],
            status="error",
        )


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[get_precise_location, get_general_area],
    # middleware: Catch tool errors
    middleware=[handle_tool_errors],
    # system_prompt: The agent's role
    system_prompt=(
        'You provide location information. Follow these steps.\n'
        '1. First look up address details with get_precise_location.\n'
        '2. Only if that fails, use get_general_area to look up the city instead of repeating the same search.\n'
        '3. If only city information is available, state that address details were not found.\n'
        '4. If both fail, say no information was found and do not guess.\n'
        'Answer concisely in English and mention that these are fictional teaching data.'
    ),
)


def show_tool_calls(messages) -> None:
    """Display tools called by the agent."""
    for message in messages:
        if message.type == "ai":
            for call in message.tool_calls:
                print(f"  [Tool call] {call['name']}({call['args']})")


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
        # Run the agent ("user": A message from the application user).
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        show_tool_calls(result["messages"])
        print(f"Answer: {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
