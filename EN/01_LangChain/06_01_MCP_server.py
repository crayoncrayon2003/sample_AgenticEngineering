import requests
from mcp.server.fastmcp import FastMCP

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8001"

#####################################
# MCP server: Expose a REST API as tools usable by AI
#####################################
# stateless_http=True: Process requests independently without retaining connection state on the server.
# json_response=True: Return ordinary JSON without a channel for server notifications.
# Without these two options, the server displays ClosedResourceError whenever a client closes a connection (processing itself succeeds).
mcp = FastMCP("store-server", host="127.0.0.1", port=8000, stateless_http=True, json_response=True)


# @mcp.tool(): Expose this function as an MCP tool. Its docstring becomes the AI's tool description.
@mcp.tool()
def store_set(key: str, value: str) -> str:
    """Save a key and value to the store. key is the key to save; value is the value to save."""
    response = requests.post(f"{API_BASE_URL}/store/set", json={"key": key, "value": value}, timeout=10)
    response.raise_for_status()
    return f"Saved key '{key}' with value '{value}'."


@mcp.tool()
def store_get(key: str) -> str:
    """Retrieve a value from the store. key is the key to look up."""
    response = requests.get(f"{API_BASE_URL}/store/get", params={"key": key}, timeout=10)
    response.raise_for_status()
    value = response.json()["value"]
    if value is None:
        return f"Key '{key}' does not exist."
    return f"Value for key '{key}': {value}"


def main():
    """Example: An LLM agent calls an MCP server, which calls a REST API. 06_01_MCP_client.py: Agent using MCP tools. 06_01_MCP_server.py: MCP server called by the agent. 06_01_MCP_api.py: REST API server called by MCP. This program: 06_01_MCP_server.py. Run: uv run python 06_01_MCP_server.py."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
