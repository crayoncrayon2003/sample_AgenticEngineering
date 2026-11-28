import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#####################################
# REST API (e.g., an internal key-value store)
#####################################
# The MCP server calls this existing API and exposes it as tools usable by AI.
app = FastAPI()
STORE = {}


class SetRequest(BaseModel):
    key: str
    value: str


# Store a key and value
@app.post("/store/set")
def set_value(request: SetRequest):
    STORE[request.key] = request.value
    return {"status": "ok"}


# Get the value for a key
@app.get("/store/get")
def get_value(key: str):
    return {"key": key, "value": STORE.get(key)}


def main():
    """Example: An LLM agent calls an MCP server, which calls a REST API. 06_01_MCP_client.py: Agent using MCP tools. 06_01_MCP_server.py: MCP server called by the agent. 06_01_MCP_api.py: REST API server called by MCP. This program: 06_01_MCP_api.py. Run: uv run python 06_01_MCP_api.py."""
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
