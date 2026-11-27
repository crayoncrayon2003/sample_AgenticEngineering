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


llm = make_llm("minimal")


###############################################################################
# ツール1 : 指定した都市の今日の天気を返すツール。ただし、天気情報は仮想データ。
###############################################################################
def get_weather(city: str) -> str:
    """指定した都市の今日の天気を返す（模擬データ）。"""
    weather = {
        "東京": "晴れ、最高気温18℃",
        "大阪": "くもり、最高気温17℃",
        "札幌": "雪、最高気温2℃",
    }
    return weather.get(city, f"{city}の天気データは登録されていません。")

###############################################################################
# ツール2 : 指定した都市への荷物の配送料金を計算するツール。ただし、料金情報は仮想データ。
###############################################################################
def get_shipping_fee(weight_kg: float, region: str) -> str:
    """荷物の重さと届け先の地域から、送料を計算する（模擬の料金表）。"""
    base = {"本州": 800, "北海道": 1200, "沖縄": 1500}[region]
    extra = max(0, int(weight_kg) - 2) * 200
    return f"送料は{base + extra}円です（{region}、{weight_kg}kg）。"


#####################################
# LLMに渡すツールの説明
#####################################
# LangChain の @tool は、関数の docstring と型ヒントからこの説明を自動で作る。ここでは自分で書く
tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "指定した都市の今日の天気を返す（模擬データ）。",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "都市名。例：東京、大阪、札幌"}},
            "required": ["city"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_shipping_fee",
        "description": "荷物の重さと届け先の地域から、送料を計算する（模擬の料金表）。",
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "荷物の重さ（kg）"},
                "region": {"type": "string", "enum": ["本州", "北海道", "沖縄"], "description": "届け先の地域"},
            },
            "required": ["weight_kg", "region"],
        },
    }},
]
# ツール名から、実行する関数を探すための対応表
tool_functions = {"get_weather": get_weather, "get_shipping_fee": get_shipping_fee}


#####################################
# エージェント（LangChain の create_agent が中で行っているループを、自分で書く）
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは案内係です。天気と送料は必ずツールで調べ、その結果だけを使って答えてください。"
    "ツールで分からないことは、分からないと伝えてください。"
    "これらは模擬データなので、実際の情報ではないことを添えてください。"
)


def run_agent(question: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    message = call_llm(llm, messages, tools=tools)

    # LLMがツール呼び出しを返す間は、ツールを実行して、もう一度LLMを呼ぶ
    while message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            # ツールの引数は、JSONの文字列で返ってくる
            args = json.loads(tool_call.function.arguments)
            result = tool_functions[tool_call.function.name](**args)
            print(f"  [ツール呼び出し] {tool_call.function.name}({args})")
            print(f"  [ツールの結果] {result}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        message = call_llm(llm, messages, tools=tools)

    # ツール呼び出しが無くなったら、それが最終回答
    messages.append(message)
    return messages


def main():
    """
    題材：天気と送料を調べられる案内エージェント（データは固定の模擬データ）。
         質問は、合計5件ある。ツールを1つ使う質問、2つ使う質問、ツールを使わない質問を含む。
    """

    # ５つの質問事項
    questions = [
        "大阪の天気を教えてください。",
        "3.5kgの荷物を北海道に送ると、送料はいくらですか？",
        "札幌の天気と、1kgの荷物を沖縄に送る送料を教えてください。",
        "那覇の天気は？",
        "ツールを使うエージェントとは何ですか？一言で教えてください。",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f"質問：{question}")
        # エージェントを実行する（ツールの呼び出しと結果は、実行中に表示する）
        messages = run_agent(question)
        print(f"回答：{messages[-1].content}")


if __name__ == "__main__":
    main()
