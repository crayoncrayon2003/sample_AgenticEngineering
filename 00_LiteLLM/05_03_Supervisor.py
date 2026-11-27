import json
import os

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


# エージェントに依頼し、最後の回答を返す（LLM → ツール実行 → LLM のループ。3人のエージェントで共通）
def ask(system_prompt: str, tools: list, message: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            result = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    return response.content


#####################################
# 専門担当1：予約担当
#####################################
def book_travel(request: str) -> str:
    """航空券やホテルの予約を処理する（模擬。実際には予約しない）。

    Args:
        request: 予約の依頼内容
    """
    print("      → [予約担当のツール] 予約処理を模擬しました。")
    return f"予約の依頼「{request}」を受け付けました（模擬処理のため、実際の予約はしていません）。"


# system_prompt : 予約担当の役割
system_prompt_booking = (
    "あなたは航空券とホテルの予約担当です。依頼を受けたら必ずbook_travelを使い、結果を日本語で簡潔に伝えてください。"
    "ツールの結果に無い内容（日程・候補・料金など）を付け加えないでください。"
)
tools_booking = [
    {"type": "function", "function": {
        "name": "book_travel",
        "description": "航空券やホテルの予約を処理する（模擬。実際には予約しない）。",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "予約の依頼内容"}},
            "required": ["request"],
        },
    }},
]


#####################################
# 専門担当2：情報担当
#####################################
# 情報担当が調べられる情報（模擬データ）
INFO = {
    "パリの観光名所": "エッフェル塔、ルーブル美術館、凱旋門、モンマルトルの丘。",
    "世界で最も高い山": "エベレスト（標高8,848メートル）。",
}


def search_info(topic: str) -> str:
    """一般的な情報を検索する（模擬データ）。

    Args:
        topic: 調べたい話題（選択肢の中から選ぶ）
    """
    print(f"      → [情報担当のツール] 「{topic}」を検索しました。")
    return INFO.get(topic, f"「{topic}」の情報は見つかりませんでした。")


# system_prompt : 情報担当の役割
system_prompt_info = (
    "あなたは一般的な質問の担当です。依頼を受けたら必ずsearch_infoで調べてください。"
    "検索結果だけを使って日本語で答え、見つからなければ、見つからなかったと伝えてください。"
)
tools_info = [
    {"type": "function", "function": {
        "name": "search_info",
        "description": "一般的な情報を検索する（模擬データ）。",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string", "enum": ["パリの観光名所", "世界で最も高い山"], "description": "調べたい話題（選択肢の中から選ぶ）"}},
            "required": ["topic"],
        },
    }},
]


#####################################
# 窓口担当が使うツール：専門担当に仕事を任せる
#####################################
def delegate_booking(request: str) -> str:
    """航空券やホテルの予約の依頼を、予約担当に任せる。

    Args:
        request: 予約の依頼内容
    """
    print("  → [窓口担当] 予約担当に任せます。")
    return ask(system_prompt_booking, tools_booking, request)


def delegate_info(question: str) -> str:
    """一般的な質問を、情報担当に任せる。

    Args:
        question: 質問の内容
    """
    print("  → [窓口担当] 情報担当に任せます。")
    return ask(system_prompt_info, tools_info, question)


#####################################
# 窓口担当のエージェント
#####################################
# system_prompt : 窓口担当の役割
system_prompt_coordinator = (
    "あなたは利用者の依頼を受け付ける窓口担当です。依頼の内容に応じて、専門担当に仕事を任せてください。"
    "自分の知識で直接答えないでください。\n"
    "- 航空券やホテルの予約：delegate_booking（日付などの詳細が足りなくても、必ず任せる）\n"
    "- 一般的な質問：delegate_info\n"
    "- 1つの依頼に、予約と質問の両方が含まれる場合：両方の専門担当に任せる\n"
    "- 意図がはっきりしない依頼：ツールを使わず、利用者に具体的な内容を尋ねる\n"
    "専門担当の結果を受け取ったら、1つの回答にまとめて、日本語で簡潔に伝えてください。"
    "専門担当の結果に無い内容（日程・候補・料金など）を付け加えないでください。"
    "模擬処理を、実際に完了したように説明しないでください。"
)
tools_coordinator = [
    {"type": "function", "function": {
        "name": "delegate_booking",
        "description": "航空券やホテルの予約の依頼を、予約担当に任せる。",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "予約の依頼内容"}},
            "required": ["request"],
        },
    }},
    {"type": "function", "function": {
        "name": "delegate_info",
        "description": "一般的な質問を、情報担当に任せる。",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "質問の内容"}},
            "required": ["question"],
        },
    }},
]

# ツール名から、実行する関数を探すための対応表（3人のエージェントで共通）
tool_functions = {"book_travel": book_travel, "search_info": search_info, "delegate_booking": delegate_booking, "delegate_info": delegate_info}


def main():
    """
    題材：
        窓口担当のエージェントが利用者の依頼を受け付け、内容に応じて、専門担当のエージェントに仕事を任せる。
        1つの依頼に複数の用件があれば、複数の専門担当に任せ、結果を1つの回答にまとめる。

    登場人物：
        窓口担当：利用者の依頼を受け付ける。どの専門担当に任せるかを自分で判断し、結果をまとめて答える。
        予約担当：専門担当の1人。航空券やホテルの予約を処理する。
        情報担当：専門担当の1人。観光名所などの一般的な情報を調べる。

    前提：
        依頼は、合計4件ある。
          1. 予約だけの依頼          → 窓口担当は、予約担当に任せる。
          2. 質問だけの依頼          → 窓口担当は、情報担当に任せる。
          3. 予約と質問の両方の依頼  → 窓口担当は、2人の専門担当に任せ、2つの結果を1つの回答にまとめる。
          4. 内容がはっきりしない依頼 → 窓口担当は、専門担当に任せず、利用者に内容を尋ねる。

    流れ（1件分）：
        1. 窓口担当が、依頼の内容を読み、どの専門担当に任せるかを決める。
        2. 任された専門担当が、自分のツールを使って仕事をし、結果を窓口担当に返す。
        3. 窓口担当が、専門担当の結果をまとめて、利用者に答える。
    """

    # 4件の依頼
    requests = [
        "パリのホテルを予約してください。",
        "パリの観光名所を教えてください。",
        "パリのホテルを予約して、パリの観光名所も教えてください。",
        "例の件、お願いします。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")
        # 窓口担当のエージェントを実行する
        answer = ask(system_prompt_coordinator, tools_coordinator, request)
        print(f"最終回答：{answer}")


if __name__ == "__main__":
    main()
