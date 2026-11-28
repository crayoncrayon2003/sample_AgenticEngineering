import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#####################################
# REST API（例：社内のキーバリューストア）
#####################################
# MCPサーバは、この既存のAPIを呼び出して、AIが使えるツールとして公開する
app = FastAPI()
STORE = {}


class SetRequest(BaseModel):
    key: str
    value: str


# キーと値を保存する
@app.post("/store/set")
def set_value(request: SetRequest):
    STORE[request.key] = request.value
    return {"status": "ok"}


# キーに対応する値を取得する
@app.get("/store/get")
def get_value(key: str):
    return {"key": key, "value": STORE.get(key)}


def main():
    """
    題材：LLMを含むエージェントが、MCPサーバを呼び出す。MCPサーバはREST APIを呼び出す。
         06_01_MCP_client.py : エージェント。ツールとしてMCPサーバを呼び出す
         06_01_MCP_server.py : MCPサーバ。エージェントから呼び出される。
         06_01_MCP_api.py    : REST APIサーバ。MCPサーバから呼び出される。
         本プログラム         ： 06_01_MCP_api.py
         起動方法             : uv run python 06_01_MCP_api.py 
    """
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
