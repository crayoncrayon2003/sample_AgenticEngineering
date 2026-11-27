import os
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

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
# LLMにツールを渡す
#####################################
tools = [get_weather, get_shipping_fee]
tools_by_name = {"get_weather": get_weather, "get_shipping_fee": get_shipping_fee}
# bind_tools：LLMに「使えるツールの一覧」を渡す。LLMは必要に応じてツール呼び出しを返す
llm_with_tools = llm.bind_tools(tools)

# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは案内係です。天気と送料は必ずツールで調べ、その結果だけを使って答えてください。"
    "ツールで分からないことは、分からないと伝えてください。"
    "これらは模擬データなので、実際の情報ではないことを添えてください。"
)


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# MessagesState：会話の履歴（messages）を持つ状態。返したメッセージは履歴に追加される

# LLMを呼ぶ。返事は「回答」か「ツール呼び出しの依頼」のどちらか
def node_model(state: MessagesState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# LLMに依頼されたツールを実行する
def node_tools(state: MessagesState) -> dict:
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
    return {"messages": tool_messages}


# 条件分岐：ツール呼び出しがあればツールへ。無ければ、それが最終回答なので終了
def select_next(state: MessagesState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


#####################################
# エージェント（LLMを呼ぶノードと、ツールを実行するノードを、ループでつないで自分で組む）
#####################################
#  START → model ─┬→ END（ツール呼び出しが無い）
#            ↑    │
#            └ tools ←┘（ツール呼び出しがある）
graph = StateGraph(MessagesState)
graph.add_node("model", node_model)
graph.add_node("tools", node_tools)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", select_next, ["tools", END])
graph.add_edge("tools", "model")
agent = graph.compile()


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
        messages = result["messages"]

        # どのツールが、どの引数で呼ばれたかを表示する
        show_tool_calls(messages)
        print(f"回答：{messages[-1].text}")


if __name__ == "__main__":
    main()
