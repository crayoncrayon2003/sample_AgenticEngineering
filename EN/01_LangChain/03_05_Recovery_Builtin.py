import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
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


# Fallback LLM:
# Used when the normal LLM keeps failing
# Use make_llm settings but change only the model name.
def make_fallback_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-5-mini", reasoning_effort="low", use_responses_api=True, timeout=60, max_retries=1)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-sonnet-5-5", reasoning_effort="low", max_tokens=16000, timeout=60, max_retries=1)


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

# Simulate a transient failure: Tool 2 raises a communication error on odd-numbered calls (an immediate retry succeeds).
flaky_state = {"calls": 0}


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
# Unlike 03_05_Recovery.py, raise a transient communication error (TimeoutError) on odd-numbered calls.
# Retrying the same call succeeds, demonstrating ToolRetryMiddleware's automatic retries.
@tool
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
# Built-in middleware 1: Automatically retry transient tool errors
#####################################
# retry_on: Exception types to retry; retry only transient failures (TimeoutError).
# Do not retry LookupError (missing data), because repeated attempts give the same result.
# on_failure: Behavior after retries fail; "error" propagates the exception outward to built-in middleware 2.
tool_retry = ToolRetryMiddleware(
    max_retries=2,
    retry_on=(TimeoutError,),
    on_failure="error",
    initial_delay=0.5,
)


#####################################
# Built-in middleware 2: Catch tool errors and report them to the AI
#####################################
# Equivalent to handle_tool_errors (custom @wrap_tool_call) in 03_05_Recovery.py.
# If on_error returns a string, that string is passed to the AI as a failure result.
# If it returns None, the exception propagates and stops the program, avoiding suppression of unexpected errors.
def on_tool_error(error: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(error, LookupError):
        print(f'  [Tool error] {error}')
        return f'Failed: {error}'
    return None


tool_error = ToolErrorMiddleware(on_error=on_tool_error)


#####################################
# Built-in middleware 3 and 4: Handle LLM call failures (normally absent in this example)
#####################################
# ModelRetryMiddleware: Retry failed LLM calls with increasing delays.
# ModelFallbackMiddleware: Switch to a fallback LLM if retries fail.
model_retry = ModelRetryMiddleware(max_retries=2)
model_fallback = ModelFallbackMiddleware(make_fallback_llm())


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[get_precise_location, get_general_area],
    # middleware: Earlier entries wrap later ones.
    # LLM: Retry (inner) -> fallback LLM if retries fail (outer)
    # Tools: Retry (inner) -> report to AI if retries fail (outer)
    middleware=[model_fallback, model_retry, tool_error, tool_retry],
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
    """Example: Recover from tool failures by switching methods, using the same requests as 03_05_Recovery.py. Built-in middleware handles errors and retries transient tool-2 communication errors before reporting them to the AI. Cases: 1. Address details found. 2. Details absent but city data found after retry. 3. Neither found; after retry, report missing data to the AI."""

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
