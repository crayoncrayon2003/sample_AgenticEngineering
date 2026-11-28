import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.messages import ToolMessage
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


llm = make_llm("low")


#####################################
# Runtime information: The logged-in user
#####################################
@dataclass
class LoginContext:
    user_id: str


#####################################
# Tool: Retrieve registered user information
#####################################
@tool
def get_user_profile(user_id: str) -> str:
    """Retrieve registered information for the specified user ID (mock data).

    Args:
        user_id: ID of the user to retrieve.
    """
    print(f'  [Executing tool body] Target user: {user_id}')
    profiles = {
        "user-001": {"name": 'Tanaka', "language": "Python"},
        "user-002": {"name": 'Sato', "language": "Java"},
    }
    return json.dumps(profiles.get(user_id, {"error": 'No matching user was found.'}), ensure_ascii=False)


#####################################
# Middleware: Validate arguments immediately before tool execution
#####################################
# Do not allow retrieval of information belonging to anyone other than the logged-in user.
@wrap_tool_call
def check_tool_args(request: ToolCallRequest, handler):
    logged_in_id = request.runtime.context.user_id
    requested_id = request.tool_call["args"].get("user_id")
    print(f'  [Argument check] Logged-in ID: {logged_in_id}; AI-requested ID: {requested_id}')

    if requested_id != logged_in_id:
        print('  [Argument check] Rejected. The tool body will not run.')
        # Return a rejection result to the AI without executing the tool.
        return ToolMessage(
            content='Retrieval was rejected because information for other users cannot be accessed.',
            tool_call_id=request.tool_call["id"],
        )

    print('  [Argument check] Allowed. Running the tool body.')
    # handler: Execute the tool body
    return handler(request)


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[get_user_profile],
    # middleware: Validation immediately before tool execution
    middleware=[check_tool_args],
    # context_schema: Schema for runtime information (logged-in user)
    context_schema=LoginContext,
    # system_prompt: The agent's role
    system_prompt=(
        'You provide user information. Pass the user ID written in the request directly as the get_user_profile argument. '
        'Answer concisely in English based on the tool results. '
        'If retrieval is rejected, state that it was rejected and do not guess the information.'
    ),
)


def main():
    """Example: Validate AI-selected arguments immediately before executing a tool. Allow access only to the logged-in user's data. The AI assistant has a user-profile tool; Tanaka (user-001), the logged-in user, requests information."""

    # Logged-in user
    login = LoginContext(user_id="user-001")

    # Two requests
    # Request 1: Tanaka asks the assistant for their own information (user-001) -> allowed.
    # Request 2: Tanaka asks for another user's information (user-002) -> rejected by validation.
    messages = [
        'Retrieve registered information for user ID user-001.',
        'Retrieve registered information for user ID user-002.',
    ]
    for message in messages:
        print("----------------------------------------")
        print(f'[{login.user_id}] User: {message}')
        # Run the agent, passing the logged-in user through context.
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, context=login)
        print(f"[{login.user_id}] AI：{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
