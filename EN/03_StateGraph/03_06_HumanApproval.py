import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
TICKET_DIR = os.path.join(ROOT, "outputs", "tickets")


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


###############################################################################
# Tool 1: Return basic troubleshooting steps; no human approval required
###############################################################################
@tool
def troubleshoot_issue(issue: str) -> str:
    """Return basic troubleshooting steps for the problem.

    Args:
        issue: Description of the problem.
    """
    return 'Checks: 1. Check the connection. 2. Record the displayed error message.'


###############################################################################
# Tool 2: Create a support ticket; human approval required
###############################################################################
@tool
def create_ticket(issue_type: str, details: str) -> str:
    """Create a support ticket and save it to a file.

    Args:
        issue_type: Type of problem.
        details: Detailed problem description.
    """
    os.makedirs(TICKET_DIR, exist_ok=True)
    ticket_id = f"TICKET-{uuid4().hex[:8]}"
    with open(os.path.join(TICKET_DIR, f"{ticket_id}.json"), "w", encoding="utf-8") as file:
        json.dump({"ticket_id": ticket_id, "issue_type": issue_type, "details": details}, file, ensure_ascii=False, indent=2)
    return f'Created support ticket {ticket_id}.'

#####################################
# Stage 1: Chain drafting a support ticket from the user's problem
#####################################
# Define the LLM response schema (ticket draft)
class TicketDraft(BaseModel):
    # issue_type: Problem type
    # details: Detailed problem description
    issue_type: str = Field(description='Problem type, e.g., printer authentication error')
    details: str = Field(description='Details of the problem and attempted troubleshooting')


prompt_draft = ChatPromptTemplate.from_messages([
    # system: Instructions from the application developer
    ("system", "You provide technical support for electronic devices. Draft a support ticket from the user's problem."),
    # user: A message from the application user
    ("user", "{message}"),
])
chain_draft = prompt_draft | llm.with_structured_output(TicketDraft, method="json_schema")

#####################################
# Final stage: Chain creating a response to the user
#####################################
prompt_reply = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You provide technical support for electronic devices. Write a concise English response to the user based on the result. '
        'If creation was rejected, state that the ticket was not created.'
    ),(
        # user: The reported problem and ticket result
        "user",
        '[Problem]\n{message}\n\n[Ticket result]\n{ticket_result}'
    ),
])
chain_reply = prompt_reply | llm | StrOutputParser()


#####################################
# Define graph state
#####################################
class SupportState(TypedDict, total=False):
    # message: User's reported problem
    # issue_type: Problem type in the support ticket
    # details: Detailed ticket description
    # approved: True if the reviewer approves
    # ticket_result: Support ticket result
    # reply: Response to the user
    message: str
    issue_type: str
    details: str
    approved: bool
    ticket_result: str
    reply: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Stage 1: Draft a support ticket
def node_draft(state: SupportState) -> dict:
    draft = chain_draft.invoke({"message": state["message"]})
    return {"issue_type": draft.issue_type, "details": draft.details}


# Stage 2: Pause and wait for the reviewer's decision. The value passed on resume (y/n) is returned here.
def node_approve(state: SupportState) -> dict:
    decision = interrupt({"issue_type": state["issue_type"], "details": state["details"]})
    return {"approved": decision == "y"}


# Stage 3: Create a support ticket (only reached after approval)
def node_create(state: SupportState) -> dict:
    return {"ticket_result": create_ticket.invoke({"issue_type": state["issue_type"], "details": state["details"]})}


# When rejected
def node_reject(state: SupportState) -> dict:
    return {"ticket_result": 'The reviewer rejected ticket creation.'}


# Final stage: Respond to the user
def node_reply(state: SupportState) -> dict:
    return {"reply": chain_reply.invoke({"message": state["message"], "ticket_result": state["ticket_result"]})}


# Conditional routing: Create if approved; otherwise do not create.
def select_next(state: SupportState) -> str:
    if state["approved"]:
        return "create"
    return "reject"


#####################################
# Build the graph
#####################################
# START -> draft -> approve (pause for a human decision) -> create -> reply -> END (approved)
# approve -> reject -> reply -> END (rejected)
graph = StateGraph(SupportState)
graph.add_node("draft", node_draft)
graph.add_node("approve", node_approve)
graph.add_node("create", node_create)
graph.add_node("reject", node_reject)
graph.add_node("reply", node_reply)
graph.add_edge(START, "draft")
graph.add_edge("draft", "approve")
graph.add_conditional_edges("approve", select_next, ["create", "reject"])
graph.add_edge("create", "reply")
graph.add_edge("reject", "reply")
graph.add_edge("reply", END)
# checkpointer: Save paused state (required for resuming)
app = graph.compile(checkpointer=InMemorySaver())


def main():
    """Example: Pause before creating a support ticket and request human approval; create only if approved. The AI workflow drafts tickets and creates approved ones. Tanaka reports a printer problem; a human support reviewer approves or rejects creation."""

    # thread_id: Conversation ID used to resume paused processing
    config = {"configurable": {"thread_id": "support-1"}}

    # ##################################################################
    # Tanaka's request: The assistant drafts a support ticket and pauses for approval.
    # ##################################################################
    message = "The printer's authentication error persists after restarting. Please create a support ticket to record it."
    print(f'[Tanaka] {message}')
    result = app.invoke({"message": message}, config)

    # If __interrupt__ is present, execution is paused for a human decision.
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"[Awaiting approval] Problem type: {request['issue_type']}")
        print(f"[Awaiting approval] Details: {request['details']}")

        # ##################################################################
        # Reviewer decision: Enter y to approve or n to reject.
        # ##################################################################
        answer = input('[Reviewer] Approve creating the support ticket? (y/n):').strip().lower()

        # Pass the decision through Command(resume=...) to resume. The supplied value becomes interrupt()'s return value.
        result = app.invoke(Command(resume=answer), config)

    print(f"[AI assistant] {result['reply']}")


if __name__ == "__main__":
    main()
