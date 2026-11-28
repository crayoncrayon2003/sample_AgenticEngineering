import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task

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
chain_extract = prompt_extract | llm.with_structured_output(Location, method="json_schema")

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
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# Tasks (individual workflow steps)
#####################################
@task
def extract(request: str) -> Location:
    return chain_extract.invoke({"request": request})


@task
def answer(request: str, precise: str, general: str) -> str:
    return chain_answer.invoke({"request": request, "precise": precise, "general": general})


#####################################
# Workflow (use try/except to switch methods after failure)
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # Stage 1: Extract the address and city from the request
    location = extract(request).result()
    print(f'  [Extracted address] {location.address} (city: {location.city})')

    # Stage 2: Search address details. Record errors and continue if it fails.
    general = '(not searched)'
    try:
        precise = get_precise_location.invoke({"address": location.address})
        print('  [Detailed search] Succeeded')
    except LookupError as error:
        precise = f'Failed: {error}'
        print(f'  [Detailed search] {precise}')

        # Stage 3: Only if detailed search fails, search city information as a fallback.
        try:
            general = get_general_area.invoke({"city": location.city})
            print('  [City search] Succeeded')
        except LookupError as error:
            general = f'Failed: {error}'
            print(f'  [City search] {general}')

    # Final stage: Generate an answer from search results
    return {"answer": answer(request, precise, general).result()}


def main():
    """Example: Recover from search failures by switching methods. Three requests cover an address match, a city-only match after detailed search fails, and neither an address nor a city match."""

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
        result = workflow.invoke(request)
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
