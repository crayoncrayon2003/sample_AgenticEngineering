import os
from typing import NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
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
# エージェントの状態を定義する
#####################################
class ShopState(AgentState):
    #  AgentState が持つ messages（会話の履歴）に加えて、独自の項目を追加する
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


#####################################
# ミドルウェア：応答が終わるたびに回数を数える
#####################################
@after_agent(state_schema=ShopState)
def count_turns(state: ShopState, runtime) -> dict:
    return {"turn_count": state.get("turn_count", 0) + 1}


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[add_to_cart, show_cart],
    # state_schema  : エージェントの状態の形（独自の項目 cart・turn_count を含む）
    state_schema=ShopState,
    # middleware    : 応答の前後に実行する処理
    middleware=[count_turns],
    # checkpointer  : 状態の保存先。thread_id ごとに保存される
    checkpointer=InMemorySaver(),
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは買い物かごの係です。\n"
        "- 商品の追加を頼まれたら、商品ごとに必ずadd_to_cartを呼ぶ。\n"
        "- かごの中身を聞かれたら、必ずshow_cartを呼ぶ。\n"
        "- ツールを呼ばずに「追加しました」などと答えてはいけない。\n"
        "ツールの結果に基づいて、日本語で短く答えてください。"
    ),
)


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

    # エージェントを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡す。
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = agent.get_state(config_a).values
    print(f"[thread-A] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # 2回目：同じ会話Aの続きで、牛乳をかごに入れる
    message = "牛乳も追加してください。"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = agent.get_state(config_a).values
    print(f"[thread-A] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # 3回目：同じ会話Aの続きで、かごの中身を確認する
    message = "今、かごに何が入っていますか？"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_a)
    print(f"[thread-A] AI：{result['messages'][-1].text}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = agent.get_state(config_a).values
    print(f"[thread-A] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")

    # ##################################################################
    # 会話B：会話A とは異なる別の会話で、かごの中身を確認する
    # ##################################################################
    message = "かごの中身を見せてください。"
    print(f"[thread-B] 利用者：{message}")

    # エージェントを実行する。このとき、config_bで、会話Bを示すthread_id = thread-B を渡す。
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config_b)
    print(f"[thread-B] AI：{result['messages'][-1].text}")

    # 会話Bの状態を取り出して、独自の項目（cart・turn_count）を表示する。会話Aとは別の状態なので、かごは空。
    state = agent.get_state(config_b).values
    print(f"[thread-B] 状態：cart={state.get('cart')}, turn_count={state.get('turn_count')}")


    # 状態の形は、「エージェントの状態を定義する（ShopState）」を参照。状態は checkpointer に会話ごとに保存される。

if __name__ == "__main__":
    main()
