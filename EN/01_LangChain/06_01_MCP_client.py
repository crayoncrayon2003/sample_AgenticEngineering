import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


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
# MCP server to connect to
#####################################
# MultiServerMCPClient connects to MCP servers and makes their tools usable as LangChain tools.
mcp_client = MultiServerMCPClient({
    "store": {"transport": "streamable_http", "url": MCP_SERVER_URL},
})


def show_tool_calls(messages) -> None:
    """Display tools called by the agent and their results."""
    for message in messages:
        if message.type == "ai":
            for call in message.tool_calls:
                print(f"  [Tool call] {call['name']}({call['args']})")
        if message.type == "tool":
            print(f'  [Tool result] {message.text}')


async def run() -> None:
    # Step 1: Retrieve the tool list (names, descriptions, and argument schemas) from the MCP server.
    tools = await mcp_client.get_tools()
    print(f'MCP server tools: {[tool.name for tool in tools]}')

    # Step 2: Pass retrieved tools to the agent.
    agent = create_agent(
        # model: The agent's LLM
        model=llm,
        # tools: Tools retrieved from the MCP server
        tools=tools,
        # system_prompt: The agent's role
        system_prompt=(
            'You operate a key-value store. Use store_set to save values and store_get to retrieve them. '
            'Answer concisely in English based on the tool results.'
        ),
    )

    # Step 3: Process requests sequentially (use ainvoke because MCP tools are asynchronous).
    requests = [
        'Save the value Python under the key favorite_language.',
        'Get the value for the key favorite_language.',
        'Get the value for the key favorite_food.',
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')
        result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        show_tool_calls(result["messages"])
        print(f"Answer: {result['messages'][-1].text}")


def main():
    """Example: An LLM agent calls an MCP server, which calls a REST API. 06_01_MCP_client.py: Agent using MCP tools. 06_01_MCP_server.py: MCP server called by the agent. 06_01_MCP_api.py: REST API server called by MCP. This program: 06_01_MCP_client.py. Run: uv run python 06_01_MCP_client.py."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
