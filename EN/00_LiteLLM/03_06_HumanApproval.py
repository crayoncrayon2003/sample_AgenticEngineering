import json
import os
from uuid import uuid4

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
TICKET_DIR = os.path.join(ROOT, "outputs", "tickets")


def make_llm(effort: str) -> dict:
    """Create LiteLLM model settings based on LLM_PROVIDER. LiteLLM selects the LLM using a 'provider/model' string. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """Call the LLM through LiteLLM and return its message (equivalent to LangChain's llm.invoke)."""
    response = litellm.completion(
        model=llm["model"],
        reasoning_effort=llm["reasoning_effort"],
        max_tokens=llm["max_tokens"],
        messages=messages,
        response_format=response_format,
        tools=tools,
        timeout=60,
        num_retries=1,
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """Fill named placeholders in the prompt with values (equivalent to LangChain's ChatPromptTemplate)."""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# Tool 1: Return basic troubleshooting steps; no human approval required
#####################################
def troubleshoot_issue(issue: str) -> str:
    """Return basic troubleshooting steps for the problem.

    Args:
        issue: Description of the problem.
    """
    return 'Checks: 1. Check the connection. 2. Record the displayed error message.'


#####################################
# Tool 2: Create a support ticket; human approval required
#####################################
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
# Tool descriptions sent to the LLM
#####################################
# LangChain's @tool generates this description from the docstring and type hints. Here we write a dictionary manually.
tools = [
    {"type": "function", "function": {
        "name": "troubleshoot_issue",
        "description": 'Return basic troubleshooting steps for the problem.',
        "parameters": {
            "type": "object",
            "properties": {"issue": {"type": "string", "description": 'Description of the problem'}},
            "required": ["issue"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": 'Create a support ticket and save it to a file.',
        "parameters": {
            "type": "object",
            "properties": {"issue_type": {"type": "string", "description": 'Type of problem'}, "details": {"type": "string", "description": 'Detailed problem description'}},
            "required": ["issue_type", "details"],
        },
    }},
]
tool_functions = {"troubleshoot_issue": troubleshoot_issue, "create_ticket": create_ticket}


#####################################
# Request human approval before executing a tool (equivalent to LangChain's HumanInTheLoopMiddleware).
#####################################
def run_tool(name: str, args: dict) -> str:
    if name == "create_ticket":
        print(f'[Awaiting approval] Tool: {name}')
        print(f'[Awaiting approval] Arguments: {json.dumps(args, ensure_ascii=False)}')
        # ##################################################################
        # The reviewer enters y to approve or n to reject. Do not execute the tool if rejected.
        # ##################################################################
        answer = input('[Reviewer] Approve creating the support ticket? (y/n):').strip().lower()
        if answer != "y":
            return 'The reviewer rejected ticket creation.'
    return tool_functions[name](**args)


#####################################
# Agent
#####################################
# system_prompt: The agent's role
system_prompt = (
    'You provide technical support for electronic devices. '
    'When asked to record an unresolved problem, create a support ticket with create_ticket. '
    'If creation is rejected, state that no ticket was created. Answer concisely in English.'
)

# Repeat LLM -> tool execution -> LLM until there are no tool calls (equivalent to LangChain's create_agent).
def run_agent(message: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            # Tool arguments are returned as a JSON string.
            args = json.loads(tool_call.function.arguments)
            result = run_tool(tool_call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    messages.append(response)
    return messages


def main():
    """Example: Pause before the AI creates a support ticket and request human approval. Create it if approved; otherwise do not. The AI assistant provides technical support and has a ticket tool. Tanaka reports a printer problem; a human support reviewer approves or rejects ticket creation."""

    # ##################################################################
    # Tanaka's request: The assistant attempts to create a support ticket and pauses.
    # ##################################################################
    message = "The printer's authentication error persists after restarting. Please create a support ticket to record it."
    print(f'[Tanaka] {message}')
    # Run the agent. It pauses just before ticket creation and receives the reviewer's decision as input.
    messages = run_agent(message)

    print(f'[AI assistant] {messages[-1].content}')


if __name__ == "__main__":
    main()
