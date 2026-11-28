import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[troubleshoot_issue, create_ticket],
    # middleware: Pause before create_ticket and wait for a human decision
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"create_ticket": True})],
    # checkpointer: Save paused state (required for resuming)
    checkpointer=InMemorySaver(),
    # system_prompt: The agent's role
    system_prompt=(
        'You provide technical support for electronic devices. '
        'When asked to record an unresolved problem, create a support ticket with create_ticket. '
        'If creation is rejected, state that no ticket was created. Answer concisely in English.'
    ),
)


def main():
    """Example: Pause before the AI creates a support ticket and request human approval. Create it if approved; otherwise do not. The AI assistant provides technical support and has a ticket tool. Tanaka reports a printer problem; a human support reviewer approves or rejects ticket creation."""

    # thread_id: Conversation ID used to resume paused processing
    config = {"configurable": {"thread_id": "support-1"}}

    # ##################################################################
    # Tanaka's request: The assistant attempts to create a support ticket and pauses.
    # ##################################################################
    message = "The printer's authentication error persists after restarting. Please create a support ticket to record it."
    print(f'[Tanaka] {message}')
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)

    # If __interrupt__ is present, execution is paused for a human decision.
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        for action in request["action_requests"]:
            print(f"[Awaiting approval] Tool: {action['name']}")
            print(f"[Awaiting approval] Arguments: {json.dumps(action['args'], ensure_ascii=False)}")

        # ##################################################################
        # Reviewer decision: Enter y to approve or n to reject.
        # ##################################################################
        answer = input('[Reviewer] Approve creating the support ticket? (y/n):').strip().lower()
        if answer == "y":
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject", "message": 'The reviewer rejected ticket creation.'}

        # Pass the reviewer's decision with Command(resume=...) to resume execution.
        result = agent.invoke(Command(resume={"decisions": [decision]}), config)

    print(f"[AI assistant] {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
