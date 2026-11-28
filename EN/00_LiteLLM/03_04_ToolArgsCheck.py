import json
import os
from dataclasses import dataclass

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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
# Runtime information: The logged-in user
#####################################
@dataclass
class LoginContext:
    user_id: str


#####################################
# Tool: Retrieve registered user information
#####################################
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
# Tool descriptions sent to the LLM
#####################################
# LangChain's @tool generates this description from the docstring and type hints. Here we write a dictionary manually.
tools = [
    {"type": "function", "function": {
        "name": "get_user_profile",
        "description": 'Retrieve registered information for the specified user ID (mock data).',
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": 'ID of the user to retrieve'}},
            "required": ["user_id"],
        },
    }},
]


#####################################
# Validate arguments immediately before tool execution (equivalent to LangChain's wrap_tool_call middleware).
#####################################
# Do not allow retrieval of information belonging to anyone other than the logged-in user.
def run_tool(args: dict, login: LoginContext) -> str:
    logged_in_id = login.user_id
    requested_id = args.get("user_id")
    print(f'  [Argument check] Logged-in ID: {logged_in_id}; AI-requested ID: {requested_id}')

    if requested_id != logged_in_id:
        print('  [Argument check] Rejected. The tool body will not run.')
        # Return a rejection result to the AI without executing the tool.
        return 'Retrieval was rejected because information for other users cannot be accessed.'

    print('  [Argument check] Allowed. Running the tool body.')
    return get_user_profile(**args)


#####################################
# Agent
#####################################
# system_prompt: The agent's role
system_prompt = (
    'You provide user information. Pass the user ID written in the request directly as the get_user_profile argument. '
    'Answer concisely in English based on the tool results. '
    'If retrieval is rejected, state that it was rejected and do not guess the information.'
)

# Repeat LLM -> tool execution -> LLM until there are no tool calls (equivalent to LangChain's create_agent).
def run_agent(message: str, login: LoginContext) -> list:
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
            result = run_tool(args, login)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    messages.append(response)
    return messages


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
        # Run the agent, passing the logged-in user.
        messages = run_agent(message, login)
        print(f"[{login.user_id}] AI：{messages[-1].content}")


if __name__ == "__main__":
    main()
