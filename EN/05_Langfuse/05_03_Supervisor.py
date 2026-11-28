import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langfuse import get_client
from langfuse.langchain import CallbackHandler

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
# Langfuse: Destination for execution records (traces)
#####################################
# Omitting the logging destination sends data to Langfuse Cloud. Send it to local Langfuse instead.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()


#####################################
# Specialist 1: Booking agent
#####################################
# Tasks handled by the booking agent (simulated)
@tool
def book_travel(request: str) -> str:
    """Process flight or hotel bookings (simulated; no actual reservation).

    Args:
        request: Booking request.
    """
    print('      -> [Booking tool] Simulated booking processing.')
    return f"Received booking request '{request}' (simulated processing; no actual reservation)."


agent_booking = create_agent(
    model=llm,
    tools=[book_travel],
    system_prompt=(
        'You handle flight and hotel bookings. Always use book_travel for requests and report the result briefly in English. '
        'Do not add details absent from the tool result (dates, options, prices, etc.).'
    ),
)


#####################################
# Specialist 2: Information agent
#####################################
# Information available to the information agent (mock data)
INFO = {
    'Tourist attractions in Paris': 'Eiffel Tower, Louvre Museum, Arc de Triomphe, and Montmartre.',
    "The world's highest mountain": 'Mount Everest (8,848 meters above sea level).',
}


@tool
def search_info(topic: Literal['Tourist attractions in Paris', "The world's highest mountain"]) -> str:
    """Search general information (mock data).

    Args:
        topic: Topic to look up (choose from the available options).
    """
    print(f"      -> [Information tool] Searched for '{topic}'.")
    return INFO.get(topic, f"No information was found for '{topic}'.")


agent_info = create_agent(
    model=llm,
    tools=[search_info],
    system_prompt=(
        'You handle general questions. Always look up requests using search_info. '
        'Answer in English using only search results. If nothing is found, say so.'
    ),
)


#####################################
# Tools for the coordinator: Delegate work to specialists
#####################################
@tool
def delegate_booking(request: str) -> str:
    """Delegate flight or hotel booking requests to the booking agent.

    Args:
        request: Booking request.
    """
    print('  -> [Coordinator] Delegating to the booking agent.')
    result = agent_booking.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def delegate_info(question: str) -> str:
    """Delegate general questions to the information agent.

    Args:
        question: The question.
    """
    print('  -> [Coordinator] Delegating to the information agent.')
    result = agent_info.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text


#####################################
# Coordinator agent
#####################################
agent_coordinator = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: Tools delegating work to specialists
    tools=[delegate_booking, delegate_info],
    # system_prompt: The agent's role
    system_prompt=(
        'You receive user requests as a coordinator. Delegate work to specialists according to the request. '
        'Do not answer directly from your own knowledge.\n'
        '- Flight or hotel bookings: delegate_booking (always delegate even if dates or other details are missing).\n'
        '- General questions: delegate_info\n'
        '- Requests containing both a booking and a question: Delegate to both specialists.\n'
        '- Requests with unclear intent: Ask the user for details without using tools.\n'
        'Combine specialist results into one concise English answer. '
        'Do not add details absent from specialist results (dates, options, prices, etc.). '
        'Do not describe simulated processing as actually completed.'
    ),
)


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
        result = agent_coordinator.invoke({"messages": [{"role": "user", "content": request}]}, config={"callbacks": [CallbackHandler()]})
        print(f"Final answer: {result['messages'][-1].text}")

    # Langfuse: Records are batched in the background; flush all remaining records before program exit.
    langfuse.flush()


if __name__ == "__main__":
    main()
