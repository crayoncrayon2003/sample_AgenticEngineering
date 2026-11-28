from uuid import uuid4

import requests

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:9000/"


#####################################
# Step 1: Retrieve the agent card and check its capabilities and endpoint.
#####################################
def get_agent_card() -> dict:
    response = requests.get(f"{SERVER_URL}.well-known/agent-card.json", timeout=10)
    response.raise_for_status()
    return response.json()


#####################################
# Step 2: Send a request in A2A format (JSON-RPC) and wait for completion.
#####################################
def send_request(card: dict, question: str) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": str(uuid4()),
                "parts": [{"kind": "text", "text": question}],
            },
            # blocking: Wait for task completion before receiving the result.
            "configuration": {"blocking": True, "acceptedOutputModes": ["text/plain"]},
        },
    }
    response = requests.post(card["url"], json=payload, timeout=180)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"A2A error: {body['error']}")
    return body["result"]


#####################################
# Step 3: Extract the answer text from the completed task's artifacts.
#####################################
def extract_answer(task: dict) -> str:
    if task["status"]["state"] != "completed":
        raise RuntimeError(f"Task is not complete: {task['status']}")
    texts = [part["text"] for artifact in task.get("artifacts", []) for part in artifact["parts"] if part["kind"] == "text"]
    return "\n".join(texts)


def main():
    """Example: A client calls an A2A server containing an LLM agent. 06_02_A2A_client.py: Retrieves the agent card and sends an A2A request. 06_02_A2A_server.py: Exposes an LLM agent to clients. This program: 06_02_A2A_client.py. Run: uv run python 06_02_A2A_client.py."""

    card = get_agent_card()
    print(f"Connected agent: {card['name']} ({card['description']})")
    print(f"Capabilities: {[skill['name'] for skill in card['skills']]}")

    # Two requests
    questions = [
        'What is ETL in data processing?',
        'Explain the roles of MCP and A2A in the context of AI agents.',
    ]
    for question in questions:
        print("----------------------------------------")
        print(f'Request: {question}')
        task = send_request(card, question)
        print(f"Task ID: {task['id']}")
        print(f'Answer: {extract_answer(task)}')


if __name__ == "__main__":
    main()
