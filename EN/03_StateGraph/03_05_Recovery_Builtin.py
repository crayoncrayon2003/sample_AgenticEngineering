import os

from dotenv import load_dotenv
from langchain.tools import tool
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

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


# Fallback LLM: Used when the usual LLM keeps failing (same settings as make_llm, changing only the model name).
def make_fallback_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-5-mini", reasoning_effort="low", use_responses_api=True, timeout=60, max_retries=1)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-sonnet-5-5", reasoning_effort="low", max_tokens=16000, timeout=60, max_retries=1)


llm = make_llm("low")
fallback_llm = make_fallback_llm()

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
# Stage 1: Chain extracting an address and city from a request
#####################################
# Define the LLM response schema (extracted address and city)
class Location(BaseModel):
    # address: Full street address
    # city: City name
    address: str = Field(description='Full street address, e.g., 1-2-3 Chuo, Sample City')
    city: str = Field(description='City name in the address, e.g., Sample City')


prompt_extract = ChatPromptTemplate.from_messages([
    # system: Instructions from the application developer
    ("system", 'Extract the address to look up and its city name from the request.'),
    # user: A message from the application user
    ("user", "{request}"),
])
# with_fallbacks switches to a backup LLM after the usual one fails (equivalent to LangChain's ModelFallbackMiddleware).
chain_extract = prompt_extract | llm.with_structured_output(Location, method="json_schema").with_fallbacks(
    [fallback_llm.with_structured_output(Location, method="json_schema")]
)

#####################################
# Final stage: Chain generating an answer from search results
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You provide location information. Answer briefly in English using only the supplied search results. '
        'If only city information is available, state that address details were not found. '
        'If both searches failed, state that no information was found and do not guess. '
        'Mention that these are fictional teaching data.'
    ),(
        # user: Request and search results
        "user",
        '[Request]\n{request}\n\n[Detailed search results]\n{precise}\n\n[City search results]\n{general}'
    ),
])
chain_answer = prompt_answer | llm.with_fallbacks([fallback_llm]) | StrOutputParser()


#####################################
# Define graph state
#####################################
class RecoveryState(TypedDict, total=False):
    # request: User request
    # address / city: Address and city extracted from the request
    # precise: Detailed address search result
    # precise_failed: True if detailed search failed
    # general: City search result
    # answer: Answer
    request: str
    address: str
    city: str
    precise: str
    precise_failed: bool
    general: str
    answer: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Stage 1: Extract the address and city from the request
def node_extract(state: RecoveryState) -> dict:
    location = chain_extract.invoke({"request": state["request"]})
    print(f'  [Extracted address] {location.address} (city: {location.city})')
    return {"address": location.address, "city": location.city}


# Stage 2: Search address details; record failure in state without stopping.
def node_precise(state: RecoveryState) -> dict:
    try:
        precise = get_precise_location.invoke({"address": state["address"]})
        print('  [Detailed search] Succeeded')
        return {"precise": precise, "precise_failed": False, "general": '(not searched)'}
    except LookupError as error:
        print(f'  [Detailed search] Failed: {error}')
        return {"precise": f'Failed: {error}', "precise_failed": True}


# Stage 3: Search city information as a fallback after detailed search fails.
def node_general(state: RecoveryState) -> dict:
    try:
        general = get_general_area.invoke({"city": state["city"]})
        print('  [City search] Succeeded')
    except LookupError as error:
        general = f'Failed: {error}'
        print(f'  [City search] {general}')
    return {"general": general}


# Final stage: Generate an answer from search results
def node_answer(state: RecoveryState) -> dict:
    answer = chain_answer.invoke({"request": state["request"], "precise": state["precise"], "general": state["general"]})
    return {"answer": answer}


# Conditional routing: Go to city search if detailed search fails.
def select_next(state: RecoveryState) -> str:
    if state["precise_failed"]:
        return "general"
    return "answer"


#####################################
# Build the graph
#####################################
# RetryPolicy automatically retries failed nodes with increasing delays (LangGraph feature).
# LLM nodes: Retry communication errors and similar failures (equivalent to LangChain's ModelRetryMiddleware).
# City-search node: Retry only transient failures (TimeoutError), equivalent to LangChain's ToolRetryMiddleware.
# Do not retry LookupError (missing data), because repeated attempts give the same result.
model_retry = RetryPolicy(max_attempts=3)
search_retry = RetryPolicy(max_attempts=3, initial_interval=0.5, retry_on=TimeoutError)

# START -> extract -> precise -> answer -> END (detailed search succeeded)
# precise -> general -> answer -> END (detailed search failed; use fallback)
graph = StateGraph(RecoveryState)
graph.add_node("extract", node_extract, retry_policy=model_retry)
graph.add_node("precise", node_precise)
graph.add_node("general", node_general, retry_policy=search_retry)
graph.add_node("answer", node_answer, retry_policy=model_retry)
graph.add_edge(START, "extract")
graph.add_edge("extract", "precise")
graph.add_conditional_edges("precise", select_next, ["general", "answer"])
graph.add_edge("general", "answer")
graph.add_edge("answer", END)
app = graph.compile()


def main():
    """Example: Recover from search failures. Three requests cover a detailed match, a city-only match after a transient-error retry, and no match after retry. Unlike 03_05_Recovery.py, automatically retry transient city-search errors and configure retries and fallback LLMs for model failures."""

    # Three requests
    requests = [
        'Look up location information for 1-2-3 Chuo, Sample City.',
        'Look up location information for 9-9-9 Chuo, Sample City.',
        'Look up location information for 9-9-9 Chuo, Unregistered City.',
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')
        # Run address extraction -> detailed search -> city search if needed -> answer together.
        result = app.invoke({"request": request})
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
