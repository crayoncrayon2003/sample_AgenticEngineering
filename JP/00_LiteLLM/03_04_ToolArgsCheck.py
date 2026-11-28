import json
import os
from dataclasses import dataclass

import litellm
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic


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
# 実行時に渡す情報：ログイン中の利用者
#####################################
@dataclass
class LoginContext:
    user_id: str


#####################################
# ツール：利用者の登録情報を取得する
#####################################
def get_user_profile(user_id: str) -> str:
    """指定した利用者IDの登録情報を取得する（模擬データ）。

    Args:
        user_id: 取得する利用者のID
    """
    print(f"  [ツール本体を実行] 取得対象：{user_id}")
    profiles = {
        "user-001": {"name": "田中", "language": "Python"},
        "user-002": {"name": "佐藤", "language": "Java"},
    }
    return json.dumps(profiles.get(user_id, {"error": "該当する利用者がいません。"}), ensure_ascii=False)


#####################################
# LLMに渡すツールの説明
#####################################
# LangChain の @tool は、関数の docstring と型ヒントからこの説明を自動で作る。ここでは辞書で書く
tools = [
    {"type": "function", "function": {
        "name": "get_user_profile",
        "description": "指定した利用者IDの登録情報を取得する（模擬データ）。",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "取得する利用者のID"}},
            "required": ["user_id"],
        },
    }},
]


#####################################
# ツールを実行する直前に、引数を検査する（LangChain のミドルウェア wrap_tool_call に当たる）
#####################################
# ログイン中の利用者以外の情報は、取得させない
def run_tool(args: dict, login: LoginContext) -> str:
    logged_in_id = login.user_id
    requested_id = args.get("user_id")
    print(f"  [引数の検査] ログイン中のID：{logged_in_id}、AIが指定したID：{requested_id}")

    if requested_id != logged_in_id:
        print("  [引数の検査] 拒否。ツール本体は実行しません。")
        # ツールを実行せず、「拒否した」という結果をAIに返す
        return "ログイン中の利用者以外の情報は取得できないため、取得を拒否しました。"

    print("  [引数の検査] 許可。ツール本体を実行します。")
    return get_user_profile(**args)


#####################################
# エージェント
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは利用者情報を案内する係です。依頼に書かれた利用者IDを、そのままget_user_profileの引数にしてください。"
    "ツールの結果に基づいて、日本語で簡潔に答えてください。"
    "取得が拒否された場合は、拒否されたことを伝え、情報を推測しないでください。"
)

# LLM → ツール実行 → LLM を、ツール呼び出しが無くなるまで繰り返す（LangChain の create_agent に当たる）
def run_agent(message: str, login: LoginContext) -> list:
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
            result = run_tool(args, login)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    messages.append(response)
    return messages


def main():
    """
    題材：ツールを実行する直前に、AIが指定した引数を検査する。ログイン中の利用者以外の情報は取得させない。

    登場人物：
      AIアシスタント          ：このプログラムのエージェント。利用者情報を取得するツールを持つ
      田中さん（ID：user-001）：ログイン中の利用者。AIアシスタントへ、利用者情報の取得を依頼する
    """

    # ログイン中の利用者
    login = LoginContext(user_id="user-001")

    # 2件の依頼
    #  依頼1：田中さんは、AIアシスタントへ、自分（user-001）の情報の取得を依頼する → 許可される
    #  依頼2：田中さんは、AIアシスタントへ、他人（user-002）の情報の取得を依頼する → 検査で拒否される
    messages = [
        "利用者ID user-001 の登録情報を取得してください。",
        "利用者ID user-002 の登録情報を取得してください。",
    ]
    for message in messages:
        print("----------------------------------------")
        print(f"[{login.user_id}] 利用者：{message}")
        # エージェントを実行する。ログイン中の利用者を渡す
        messages = run_agent(message, login)
        print(f"[{login.user_id}] AI：{messages[-1].content}")


if __name__ == "__main__":
    main()
