import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task
from pydantic import BaseModel, Field

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


llm = make_llm("minimal")

#####################################
# Define the LLM response schema
#####################################
class Route(BaseModel):
    # reason: Why this handler was selected.
    # destination: The selected handler; one of "expense", "it", or "unclear".
    # Order the fields as reason, then destination.
    # The LLM outputs fields in order. Asking for a reason first helps it select a handler consistent with that reason.
    reason: str = Field(description='The inquiry and the reasoning for the appropriate handler (one sentence in English)')
    destination: Literal["expense", "it", "unclear"] = Field(description='expense: Expense reimbursement; it: PC or network issues; unclear: Cannot determine')


prompt_route = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You are the receptionist for an internal help desk. Read the inquiry and select one handler. If you cannot identify what the inquiry is about, select unclear.',
    ),(
        # user: A message from the application user
        "user",
        "{request}"
    ),
])
chain_route = prompt_route | llm.with_structured_output(Route, method="json_schema")

#####################################
# Chains for each handler
#####################################
def make_chain_handler(role: str):
    prompt_handler = ChatPromptTemplate.from_messages([
        (
            # system: Instructions from the application developer
            "system",
            role + 'Answer in English in no more than three sentences.'
        ),(
            # user: A message from the application user
            "user",
            "{request}"
        ),
    ])
    chain_handler = prompt_handler | llm | StrOutputParser()
    return chain_handler


chain_handlers = {
    "expense": make_chain_handler('You handle accounting. Explain the expense reimbursement procedure. '
                                  'For anything unclear in the internal rules, recommend checking with the accounting department.'),
    "it"     : make_chain_handler('You handle IT. Explain checks the user can perform.'),
    "unclear": make_chain_handler('You are the receptionist. Since the request is unclear, '
                                  'ask specifically what the user needs help with.'),
}


#####################################
# Tasks (individual workflow steps)
#####################################
# Stage 1: Select a handler
@task
def route(request: str) -> Route:
    return chain_route.invoke({"request": request})


# Stage 2: Answer using the selected handler's chain
@task
def handle(destination: str, request: str) -> str:
    return chain_handlers[destination].invoke({"request": request})


#####################################
# Workflow (describe the control flow in ordinary Python)
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # Calling a task immediately returns a future. Use .result() to retrieve the result.
    result_step1 = route(request).result()
    # Use the first stage's destination to choose the second stage's handler.
    result_step2 = handle(result_step1.destination, request).result()
    return {"destination": result_step1.destination, "reason": result_step1.reason, "answer": result_step2}


def main():
    """Example: Route three internal help desk inquiries to expense, IT, or unclear."""

    requests = [
        'I want to claim a taxi fare from a business trip, but I lost the receipt.',
        "I cannot connect to the meeting room's Wi-Fi.",
        'What is happening with that matter?',
    ]

    for request in requests:
        print("----------------------------------------")
        print(f'Inquiry: {request}')

        # Run both stages together: Select a handler, then let it answer.
        result = workflow.invoke(request)
        print(f"Selected handler: {result['destination']}")
        print(f"Routing reason: {result['reason']}")
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
