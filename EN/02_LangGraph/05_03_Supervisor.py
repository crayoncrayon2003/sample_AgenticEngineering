import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
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
# Coordinator: Chain deciding which specialists to delegate to
#####################################
# Define the LLM response schema (coordinator decision)
class Assignment(BaseModel):
    # reason: Reason for the decision
    # need_booking: True to delegate to the booking agent
    # need_info: True to delegate to the information agent
    # Put reasons before decisions so the model explains before deciding.
    reason: str = Field(description='Reason for the decision (one sentence in English)')
    need_booking: bool = Field(description='true if the request includes flight or hotel booking, even if dates or other details are missing')
    need_info: bool = Field(description='true if the request includes a general question, such as tourist attractions')


prompt_assign = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You coordinate user requests. Decide whether a request includes a booking, a question, or both. '
        'If both are present, set both to true. If intent is unclear, set both to false.'
    ),(
        # user: A message from the application user
        "user",
        "{request}"
    ),
])
chain_assign = prompt_assign | llm.with_structured_output(Assignment, method="json_schema")

#####################################
# Coordinator: Chain combining specialists' results into an answer
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You coordinate user requests. Combine specialist results into one concise English answer. '
        'If there are no specialist results, ask specifically what the user needs help with. '
        'Do not add dates, options, prices, or other details absent from results, or describe simulations as completed real actions.'
    ),(
        # user: Request and specialist results
        "user",
        '[Request]\n{request}\n\n[Booking result]\n{booking}\n\n[Information result]\n{info}'
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


# Ask the specialist agent and return only its final answer.
def ask(agent, request: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


#####################################
# Tasks (individual workflow steps)
#####################################
@task
def assign(request: str) -> Assignment:
    return chain_assign.invoke({"request": request})


@task
def task_booking(request: str) -> str:
    print('  -> [Coordinator] Delegating to the booking agent.')
    return ask(agent_booking, request)


@task
def task_info(request: str) -> str:
    print('  -> [Coordinator] Delegating to the information agent.')
    return ask(agent_info, request)


@task
def answer(request: str, booking: str, info: str) -> str:
    return chain_answer.invoke({"request": request, "booking": booking, "info": info})


#####################################
# Workflow: Coordinator decides -> specialists (both concurrently if needed) -> coordinator combines
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # Stage 1: The coordinator decides which specialists to use.
    assignment = assign(request).result()
    print(f'  [Coordinator decision] Booking: {assignment.need_booking}; information: {assignment.need_info} ({assignment.reason})')

    # Stage 2: Delegate to required specialists. If both are needed, start both before .result() so they run concurrently.
    future_booking = task_booking(request) if assignment.need_booking else None
    future_info = task_info(request) if assignment.need_info else None
    booking = future_booking.result() if future_booking else '(not delegated)'
    info = future_info.result() if future_info else '(not delegated)'

    # Stage 3: The coordinator combines specialist results and answers.
    return {"answer": answer(request, booking, info).result()}


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
        # Run coordinator decision -> specialists -> coordinator summary.
        result = workflow.invoke(request)
        print(f"Final answer: {result['answer']}")


if __name__ == "__main__":
    main()
