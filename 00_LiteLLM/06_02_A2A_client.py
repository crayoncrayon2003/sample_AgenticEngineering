from uuid import uuid4

import requests

# --- 設定 ---
SERVER_URL = "http://127.0.0.1:9000/"


#####################################
# Step1 エージェントカードを取得し、相手の能力と接続先を確かめる
#####################################
def get_agent_card() -> dict:
    response = requests.get(f"{SERVER_URL}.well-known/agent-card.json", timeout=10)
    response.raise_for_status()
    return response.json()


#####################################
# Step2 A2Aの形式（JSON-RPC）で依頼を送り、完了を待つ
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
            # blocking：タスクが完了するまで待ってから、結果を受け取る
            "configuration": {"blocking": True, "acceptedOutputModes": ["text/plain"]},
        },
    }
    response = requests.post(card["url"], json=payload, timeout=180)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"A2Aのエラー：{body['error']}")
    return body["result"]


#####################################
# Step3 完了したタスクの成果物から、回答の文章を取り出す
#####################################
def extract_answer(task: dict) -> str:
    if task["status"]["state"] != "completed":
        raise RuntimeError(f"タスクが完了していません：{task['status']}")
    texts = [part["text"] for artifact in task.get("artifacts", []) for part in artifact["parts"] if part["kind"] == "text"]
    return "\n".join(texts)


def main():
    """
    題材：クライアントが、A2Aサーバ（LLMを含むエージェント）を呼び出す。
         06_02_A2A_client.py : クライアント。エージェントカードを取得し、A2Aでエージェントに依頼する。
         06_02_A2A_server.py : A2Aサーバ。LLMを含むエージェントを公開し、クライアントから呼び出される。
         本プログラム         ： 06_02_A2A_client.py
         起動方法             : uv run python 06_02_A2A_client.py
    """

    card = get_agent_card()
    print(f"接続先：{card['name']}（{card['description']}）")
    print(f"能力：{[skill['name'] for skill in card['skills']]}")

    # 2件の依頼
    questions = [
        "データ処理におけるETLとは何ですか？",
        "AIエージェントの文脈で、MCPとA2Aの役割の違いを説明してください。",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f"依頼：{question}")
        task = send_request(card, question)
        print(f"タスクID：{task['id']}")
        print(f"回答：{extract_answer(task)}")


if __name__ == "__main__":
    main()
