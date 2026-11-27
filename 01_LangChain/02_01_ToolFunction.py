import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic


def make_llm(effort: str):
    """LLM_PROVIDER に応じて、OpenAI または Claude のチャットモデルを作る。

    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL") or "gpt-5-nano",
            reasoning_effort=effort,
            use_responses_api=True,
            timeout=60,
            max_retries=1,
        )
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


llm = make_llm("minimal")


###############################################################################
# ツール1 : 指定した都市の今日の天気を返すツール。ただし、天気情報は仮想データ。
###############################################################################
@tool
def get_weather(city: str) -> str:
    """指定した都市の今日の天気を返す（模擬データ）。

    Args:
        city: 都市名。例：東京、大阪、札幌
    """
    weather = {
        "東京": "晴れ、最高気温18℃",
        "大阪": "くもり、最高気温17℃",
        "札幌": "雪、最高気温2℃",
    }
    return weather.get(city, f"{city}の天気データは登録されていません。")

###############################################################################
# ツール2 : 指定した都市への荷物の配送料金を計算するツール。ただし、料金情報は仮想データ。
###############################################################################
@tool
def get_shipping_fee(weight_kg: float, region: Literal["本州", "北海道", "沖縄"]) -> str:
    """荷物の重さと届け先の地域から、送料を計算する（模擬の料金表）。

    Args:
        weight_kg: 荷物の重さ（kg）
        region: 届け先の地域
    """
    base = {"本州": 800, "北海道": 1200, "沖縄": 1500}[region]
    extra = max(0, int(weight_kg) - 2) * 200
    return f"送料は{base + extra}円です（{region}、{weight_kg}kg）。"


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[get_weather, get_shipping_fee],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは案内係です。天気と送料は必ずツールで調べ、その結果だけを使って答えてください。"
        "ツールで分からないことは、分からないと伝えてください。"
        "これらは模擬データなので、実際の情報ではないことを添えてください。"
    ),
)


def show_tool_calls(messages) -> None:
    """エージェントが呼んだツールと、その結果を表示する。"""
    for message in messages:
        for call in getattr(message, "tool_calls", []):
            print(f"  [ツール呼び出し] {call['name']}({call['args']})")
        if message.type == "tool":
            print(f"  [ツールの結果] {message.text}")


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
        # エージェントを実行する（"user"：アプリを使う利用者の発言）
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        # どのツールが、どの引数で呼ばれたかを表示する
        show_tool_calls(result["messages"])
        print(f"回答：{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
