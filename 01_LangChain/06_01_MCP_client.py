import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"


def make_llm(effort: str):
    """LLM_PROVIDER に応じて、OpenAI または Claude のチャットモデルを作る。

    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
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
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


llm = make_llm("low")

#####################################
# 接続するMCPサーバ
#####################################
# MultiServerMCPClient：MCPサーバに接続し、サーバのツールを LangChain のツールとして使えるようにする
mcp_client = MultiServerMCPClient({
    "store": {"transport": "streamable_http", "url": MCP_SERVER_URL},
})


def show_tool_calls(messages) -> None:
    """エージェントが呼んだツールと、その結果を表示する。"""
    for message in messages:
        if message.type == "ai":
            for call in message.tool_calls:
                print(f"  [ツール呼び出し] {call['name']}({call['args']})")
        if message.type == "tool":
            print(f"  [ツールの結果] {message.text}")


async def run() -> None:
    # Step1 MCPサーバから、ツールの一覧（名前・説明・引数の形）を取得する
    tools = await mcp_client.get_tools()
    print(f"MCPサーバのツール：{[tool.name for tool in tools]}")

    # Step2 取得したツールを、エージェントに渡す
    agent = create_agent(
        # model　　　　　: エージェントに設定するLLM
        model=llm,
        # tools         : MCPサーバから取得したツール
        tools=tools,
        # system_prompt : エージェントに設定する役割
        system_prompt=(
            "あなたはキーバリューストアの操作係です。保存の依頼には store_set、取得の依頼には store_get を使ってください。"
            "ツールの結果に基づいて、日本語で簡潔に答えてください。"
        ),
    )

    # Step3 依頼を順番に実行する（MCPのツールは非同期で動くため、ainvoke を使う）
    requests = [
        "キー favorite_language に、値 Python を保存してください。",
        "キー favorite_language の値を取得してください。",
        "キー favorite_food の値を取得してください。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": request}]})
        show_tool_calls(result["messages"])
        print(f"回答：{result['messages'][-1].text}")


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
