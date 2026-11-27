import requests
from mcp.server.fastmcp import FastMCP

# --- 設定 ---
API_BASE_URL = "http://127.0.0.1:8001"

#####################################
# MCPサーバ：REST APIを、AIが使えるツールとして公開する
#####################################
#  stateless_http=True：依頼ごとに独立して処理し、サーバ側に接続の状態を残さない
#  json_response=True ：結果を通常のJSONで返す（サーバからの通知用の通信路を使わない）
#  この2つが無いと、クライアントが接続を閉じるたびに、サーバに ClosedResourceError が表示される（処理は成功している）
mcp = FastMCP("store-server", host="127.0.0.1", port=8000, stateless_http=True, json_response=True)


# @mcp.tool()：この関数を、MCPのツールとして公開する。docstring が、AIへのツールの説明になる
@mcp.tool()
def store_set(key: str, value: str) -> str:
    """キーと値をストアに保存する。keyは保存するキー、valueは保存する値。"""
    response = requests.post(f"{API_BASE_URL}/store/set", json={"key": key, "value": value}, timeout=10)
    response.raise_for_status()
    return f"キー '{key}' に値 '{value}' を保存しました。"


@mcp.tool()
def store_get(key: str) -> str:
    """キーに対応する値をストアから取得する。keyは取得したいキー。"""
    response = requests.get(f"{API_BASE_URL}/store/get", params={"key": key}, timeout=10)
    response.raise_for_status()
    value = response.json()["value"]
    if value is None:
        return f"キー '{key}' は存在しません。"
    return f"キー '{key}' の値：{value}"


def main():
    """
    題材：LLMを含むエージェントが、MCPサーバを呼び出す。MCPサーバはREST APIを呼び出す。
         06_01_MCP_client.py : エージェント。ツールとしてMCPサーバを呼び出す
         06_01_MCP_server.py : MCPサーバ。エージェントから呼び出される。
         06_01_MCP_api.py    : REST APIサーバ。MCPサーバから呼び出される。
         本プログラム         ： 06_01_MCP_server.py
         起動方法             : uv run python 06_01_MCP_server.py
    """
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
