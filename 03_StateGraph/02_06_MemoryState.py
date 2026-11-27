import os
from typing import NotRequired

from dotenv import load_dotenv
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

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


llm = make_llm("low")  # minimalだと、ツールを呼ばずに「追加しました」と答えることがある


#####################################
# グラフの状態を定義する
#####################################
class ShopState(MessagesState):
    #  MessagesState が持つ messages（会話の履歴）に加えて、独自の項目を追加する
    #  cart      ： 買い物かごの中身
    #  turn_count： エージェントが応答した回数
    cart: NotRequired[list[str]]
    turn_count: NotRequired[int]


#####################################
# ツール：状態を読む・更新する
#####################################
@tool
def add_to_cart(item: str, runtime: ToolRuntime) -> Command:
    """商品を買い物かごに追加する。

    Args:
        item: 追加する商品名
    """
    cart = runtime.state.get("cart", []) + [item]
    # Command で、状態の項目とツールの結果メッセージを同時に更新する
    return Command(update={
        "cart": cart,
        "messages": [ToolMessage(f"{item}を追加しました。かご：{cart}", tool_call_id=runtime.tool_call_id)],
    })


@tool
def show_cart(runtime: ToolRuntime) -> str:
    """買い物かごの中身を返す。"""
    cart = runtime.state.get("cart", [])
    return f"かごの中身：{cart}" if cart else "かごは空です。"


llm_with_tools = llm.bind_tools([add_to_cart, show_cart])

# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは買い物かごの係です。\n"
    "- 商品の追加を頼まれたら、商品ごとに必ずadd_to_cartを呼ぶ。\n"
    "- かごの中身を聞かれたら、必ずshow_cartを呼ぶ。\n"
    "- ツールを呼ばずに「追加しました」などと答えてはいけない。\n"
    "ツールの結果に基づいて、日本語で短く答えてください。"
)


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# LLMを呼ぶ
def node_model(state: ShopState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# 応答が終わるたびに回数を数える
def node_count_turns(state: ShopState) -> dict:
    return {"turn_count": state.get("turn_count", 0) + 1}


# 条件分岐：ツール呼び出しがあればツールへ。無ければ、回数を数えて終了
def select_next(state: ShopState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "count_turns"


#####################################
# グラフを組み立てる
#####################################
#  START → model ─┬→ count_turns → END（ツール呼び出しが無い）
#            ↑    │
#            └ tools ←┘（ツール呼び出しがある）
graph = StateGraph(ShopState)
graph.add_node("model", node_model)
# ToolNode：ツールを実行するノード。ツールに runtime（状態など）を渡し、Command による状態の更新も反映する
graph.add_node("tools", ToolNode([add_to_cart, show_cart]))
graph.add_node("count_turns", node_count_turns)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", select_next, ["tools", "count_turns"])
graph.add_edge("tools", "model")
graph.add_edge("count_turns", END)
# checkpointer を渡すと、thread_id ごとに状態が保存される
app = graph.compile(checkpointer=InMemorySaver())


def main():
    """
    題材：買い物かごを管理するエージェント。かごの中身と、応答した回数を状態に保存する。
         会話A（thread-A）で3回、会話B（thread-B）で1回依頼する。
    """

    # thread_id：どの会話の続きかを表すID。会話ごとに別のIDを使う
    config_a = {"configurable": {"thread_id": "thread-A"}}
    config_b = {"configurable": {"thread_id": "thread-B"}}

    # ##################################################################
    # 会話A：1回目にりんご、2回目に牛乳をかごに入れて、3回目にかごの中身を確認する
    # ##################################################################
    # 1回目：りんごをかごに入れる
    message = "りんごをかごに入れてください。"
    print(f"[thread-A] 利用者：{message}")

    # グラフを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡す。
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = app.get_state(config_a).values
    print(f"[thread-A] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # 2回目：同じ会話Aの続きで、牛乳をかごに入れる
    message = "牛乳も追加してください。"
    print(f"[thread-A] 利用者：{message}")

    # グラフを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = app.get_state(config_a).values
    print(f"[thread-A] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # 3回目：同じ会話Aの続きで、かごの中身を確認する
    message = "今、かごに何が入っていますか？"
    print(f"[thread-A] 利用者：{message}")

    # グラフを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = app.get_state(config_a).values
    print(f"[thread-A] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")

    # ##################################################################
    # 会話B：会話A とは異なる別の会話で、かごの中身を確認する
    # ##################################################################
    message = "かごの中身を見せてください。"
    print(f"[thread-B] 利用者：{message}")

    # グラフを実行する。このとき、config_bで、会話Bを示すthread_id = thread-B を渡す。
    result = app.invoke({"messages": [{"role": "user", "content": message}]}, config_b)
    print(f"[thread-B] AI：{result['messages'][-1].text}")

    # 会話Bの状態を取り出して、独自の項目（cart・turn_count）を表示する。会話Aとは別の状態なので、かごは空。
    state = app.get_state(config_b).values
    print(f"[thread-B] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # 状態の形は、「グラフの状態を定義する（ShopState）」を参照。状態は checkpointer に会話ごとに保存される。

if __name__ == "__main__":
    main()
