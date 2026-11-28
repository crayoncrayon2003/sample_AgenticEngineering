import json
import os
from uuid import uuid4

import litellm
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
TICKET_DIR = os.path.join(ROOT, "outputs", "tickets")


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
# ツール1 : 基本的な確認手順を返す。人間の承認は不要
#####################################
def troubleshoot_issue(issue: str) -> str:
    """問題に対する、基本的な確認手順を返す。

    Args:
        issue: 問題の内容
    """
    return "確認手順：1. 接続状態を確認する。2. 表示されたエラーメッセージを記録する。"


#####################################
# ツール2 : 問い合わせ票を作る。人間の承認は必要
#####################################
def create_ticket(issue_type: str, details: str) -> str:
    """問い合わせ票を作成し、ファイルに保存する。

    Args:
        issue_type: 問題の種類
        details: 問題の詳しい内容
    """
    os.makedirs(TICKET_DIR, exist_ok=True)
    ticket_id = f"TICKET-{uuid4().hex[:8]}"
    with open(os.path.join(TICKET_DIR, f"{ticket_id}.json"), "w", encoding="utf-8") as file:
        json.dump({"ticket_id": ticket_id, "issue_type": issue_type, "details": details}, file, ensure_ascii=False, indent=2)
    return f"問い合わせ票 {ticket_id} を作成しました。"


#####################################
# LLMに渡すツールの説明
#####################################
# LangChain の @tool は、関数の docstring と型ヒントからこの説明を自動で作る。ここでは辞書で書く
tools = [
    {"type": "function", "function": {
        "name": "troubleshoot_issue",
        "description": "問題に対する、基本的な確認手順を返す。",
        "parameters": {
            "type": "object",
            "properties": {"issue": {"type": "string", "description": "問題の内容"}},
            "required": ["issue"],
        },
    }},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "問い合わせ票を作成し、ファイルに保存する。",
        "parameters": {
            "type": "object",
            "properties": {"issue_type": {"type": "string", "description": "問題の種類"}, "details": {"type": "string", "description": "問題の詳しい内容"}},
            "required": ["issue_type", "details"],
        },
    }},
]
tool_functions = {"troubleshoot_issue": troubleshoot_issue, "create_ticket": create_ticket}


#####################################
# ツールを実行する前に、人間の承認を求める（LangChain の HumanInTheLoopMiddleware に当たる）
#####################################
def run_tool(name: str, args: dict) -> str:
    if name == "create_ticket":
        print(f"[承認待ち] ツール：{name}")
        print(f"[承認待ち] 引数　：{json.dumps(args, ensure_ascii=False)}")
        # ##################################################################
        # 担当者の判断：承認（y）か却下（n）を入力する。却下なら、ツールを実行しない
        # ##################################################################
        answer = input("[担当者] 問い合わせ票の作成を承認しますか？（y/n）：").strip().lower()
        if answer != "y":
            return "担当者が作成を却下しました。"
    return tool_functions[name](**args)


#####################################
# エージェント
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは電子機器の技術サポート係です。"
    "解決しない問題を記録するよう頼まれたら、create_ticketで問い合わせ票を作成してください。"
    "作成が却下された場合は、作成しなかったことを伝えてください。日本語で簡潔に答えてください。"
)

# LLM → ツール実行 → LLM を、ツール呼び出しが無くなるまで繰り返す（LangChain の create_agent に当たる）
def run_agent(message: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            # ツールの引数は、JSONの文字列で返ってくる
            args = json.loads(tool_call.function.arguments)
            result = run_tool(tool_call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    messages.append(response)
    return messages


def main():
    """
    題材：AIが問い合わせ票を作る前に、処理を止めて人間の担当者に承認を求める。
         承認されたら作成し、却下されたら作成しない。

    登場人物：
      AIアシスタント：このプログラムのエージェント。技術サポートを行い、問い合わせ票を作るツールを持つ
      田中さん      ：田中さんは、AIアシスタントへ、プリンターの不具合を相談する
      担当者        ：人間のサポート担当者。AIアシスタントが問い合わせ票を作る前に、承認または却下する
    """

    # ##################################################################
    # 田中さんの相談：AIアシスタントは問い合わせ票を作ろうとして、止まる
    # ##################################################################
    message = "プリンターの認証エラーが再起動しても直りません。問い合わせ票を作って記録してください。"
    print(f"[田中さん] {message}")
    # エージェントを実行する。問い合わせ票を作る直前で止まり、担当者の判断を入力で受け取る
    messages = run_agent(message)

    print(f"[AIアシスタント] {messages[-1].content}")


if __name__ == "__main__":
    main()
