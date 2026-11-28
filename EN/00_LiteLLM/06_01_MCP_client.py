import asyncio
import json
import os

import litellm
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


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
# Convert MCP tools into tool descriptions for LiteLLM.
#####################################
# Convert MCP tool names, descriptions, and inputSchema into the format sent to the LLM.
def to_litellm_tool(mcp_tool) -> dict:
    return {"type": "function", "function": {
        "name": mcp_tool.name,
        "description": mcp_tool.description or mcp_tool.name,
        "parameters": mcp_tool.inputSchema,
    }}


async def run() -> None:
    # Connect to the MCP server
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Step 1: Retrieve the tool list (names, descriptions, and argument schemas) from the MCP server.
            result_tools = await session.list_tools()
            tools = [to_litellm_tool(tool) for tool in result_tools.tools]
            print(f'MCP server tools: {[tool.name for tool in result_tools.tools]}')

            # Step 2: Pass the retrieved tools to the LLM.
            # system_prompt: The agent's role
            system_prompt = (
                'You operate a key-value store. Use store_set to save values and store_get to retrieve them. '
                'Answer concisely in English based on the tool results.'
            )

            # Step 3: Process requests sequentially (LLM -> MCP tool execution -> LLM loop).
            requests = [
                'Save the value Python under the key favorite_language.',
                'Get the value for the key favorite_language.',
                'Get the value for the key favorite_food.',
            ]
            for request in requests:
                print("----------------------------------------")
                print(f'Request: {request}')
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": request}]
                response = call_llm(llm, messages, tools=tools)
                while response.tool_calls:
                    messages.append(response)
                    for tool_call in response.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        print(f'  [Tool call] {tool_call.function.name}({args})')
                        # Ask the MCP server to execute a tool.
                        result_call = await session.call_tool(tool_call.function.name, arguments=args)
                        text = "\n".join(block.text for block in result_call.content if block.type == "text")
                        print(f'  [Tool result] {text}')
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": text})
                    response = call_llm(llm, messages, tools=tools)
                print(f'Answer: {response.content}')


def main():
    """Example: An LLM agent calls an MCP server, which calls a REST API. 06_01_MCP_client.py: Agent using MCP tools. 06_01_MCP_server.py: MCP server called by the agent. 06_01_MCP_api.py: REST API server called by MCP. This program: 06_01_MCP_client.py. Run: uv run python 06_01_MCP_client.py."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
