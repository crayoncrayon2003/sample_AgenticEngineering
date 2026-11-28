import asyncio
import json
import os

import litellm
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


def make_llm(effort: str) -> dict:
    """LLM_PROVIDER に応じて、LiteLLM に渡すモデルの設定を作る。

    LiteLLM は「提供元/モデル名」の文字列で、呼び出すLLMを切り替える。
    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """LiteLLM でLLMを呼び、返ってきたメッセージを返す（LangChain の llm.invoke に当たる）。"""
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
    """プロンプトの {名前} に値を入れる（LangChain の ChatPromptTemplate に当たる）。"""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# MCPのツールを、LiteLLM に渡すツールの説明に変換する
#####################################
# MCPサーバのツールの「名前・説明・引数の形（inputSchema）」を、LLMに渡す形にする
def to_litellm_tool(mcp_tool) -> dict:
    return {"type": "function", "function": {
        "name": mcp_tool.name,
        "description": mcp_tool.description or mcp_tool.name,
        "parameters": mcp_tool.inputSchema,
    }}


async def run() -> None:
    # MCPサーバに接続する
    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Step1 MCPサーバから、ツールの一覧（名前・説明・引数の形）を取得する
            result_tools = await session.list_tools()
            tools = [to_litellm_tool(tool) for tool in result_tools.tools]
            print(f"MCPサーバのツール：{[tool.name for tool in result_tools.tools]}")

            # Step2 取得したツールを、LLMに渡す
            # system_prompt : エージェントに設定する役割
            system_prompt = (
                "あなたはキーバリューストアの操作係です。保存の依頼には store_set、取得の依頼には store_get を使ってください。"
                "ツールの結果に基づいて、日本語で簡潔に答えてください。"
            )

            # Step3 依頼を順番に実行する（LLM → MCPのツール実行 → LLM のループ）
            requests = [
                "キー favorite_language に、値 Python を保存してください。",
                "キー favorite_language の値を取得してください。",
                "キー favorite_food の値を取得してください。",
            ]
            for request in requests:
                print("----------------------------------------")
                print(f"依頼：{request}")
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": request}]
                response = call_llm(llm, messages, tools=tools)
                while response.tool_calls:
                    messages.append(response)
                    for tool_call in response.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        print(f"  [ツール呼び出し] {tool_call.function.name}({args})")
                        # MCPサーバに、ツールの実行を依頼する
                        result_call = await session.call_tool(tool_call.function.name, arguments=args)
                        text = "\n".join(block.text for block in result_call.content if block.type == "text")
                        print(f"  [ツールの結果] {text}")
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": text})
                    response = call_llm(llm, messages, tools=tools)
                print(f"回答：{response.content}")


def main():
    """
    題材：LLMを含むエージェントが、MCPサーバを呼び出す。MCPサーバはREST APIを呼び出す。
         06_01_MCP_client.py : エージェント。ツールとしてMCPサーバを呼び出す
         06_01_MCP_server.py : MCPサーバ。エージェントから呼び出される。
         06_01_MCP_api.py    : REST APIサーバ。MCPサーバから呼び出される。
         本プログラム         ： 06_01_MCP_client.py
         起動方法             : uv run python 06_01_MCP_client.py
    """
    asyncio.run(run())


if __name__ == "__main__":
    main()
