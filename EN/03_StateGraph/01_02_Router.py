import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
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
# Define graph state
#####################################
class RouterState(TypedDict, total=False):
    # request: Inquiry
    # destination: Selected handler
    # reason: Reason for routing
    # answer: Handler's answer
    request: str
    destination: str
    reason: str
    answer: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Stage 1: Select a handler
def node_route(state: RouterState) -> dict:
    result_step1 = chain_route.invoke({"request": state["request"]})
    return {"destination": result_step1.destination, "reason": result_step1.reason}


# Stage 2: Nodes for each handler
def node_expense(state: RouterState) -> dict:
    return {"answer": chain_handlers["expense"].invoke({"request": state["request"]})}


def node_it(state: RouterState) -> dict:
    return {"answer": chain_handlers["it"].invoke({"request": state["request"]})}


def node_unclear(state: RouterState) -> dict:
    return {"answer": chain_handlers["unclear"].invoke({"request": state["request"]})}


# Conditional routing: Go to the node named by destination.
def select_handler(state: RouterState) -> str:
    return state["destination"]


#####################################
# Build the graph
#####################################
#  START → route ─┬→ expense ─┐
#                 ├→ it      ─┼→ END
#                 └→ unclear ─┘
graph = StateGraph(RouterState)
graph.add_node("route", node_route)
graph.add_node("expense", node_expense)
graph.add_node("it", node_it)
graph.add_node("unclear", node_unclear)
graph.add_edge(START, "route")
graph.add_conditional_edges("route", select_handler, ["expense", "it", "unclear"])
graph.add_edge("expense", END)
graph.add_edge("it", END)
graph.add_edge("unclear", END)
app = graph.compile()


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
        result = app.invoke({"request": request})
        print(f"Selected handler: {result['destination']}")
        print(f"Routing reason: {result['reason']}")
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
