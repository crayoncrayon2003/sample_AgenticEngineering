import os
from typing import Annotated

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import InjectedToolArg, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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
# ツール：かごの中身（cart）を受け取って処理する
#####################################
# InjectedToolArg：LLMには見せず、プログラムから渡す引数。かごの中身はワークフローが渡す
@tool
def add_to_cart(item: str, cart: Annotated[list[str], InjectedToolArg]) -> list[str]:
    """商品を買い物かごに追加する。

    Args:
        item: 追加する商品名
    """
    return cart + [item]


@tool
def show_cart(cart: Annotated[list[str], InjectedToolArg]) -> list[str]:
    """買い物かごの中身を返す。"""
    return cart


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
# タスク：LLMを呼ぶ
#####################################
@task
def call_model(messages: list) -> AIMessage:
    return llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)


#####################################
# エージェント（状態を、自分で受け取って保存する）
#####################################
# checkpointer を渡すと、thread_id ごとに「前回の保存値（previous）」が保存される
@entrypoint(checkpointer=InMemorySaver())
def agent(message: str, *, previous: dict | None = None) -> dict:
    # previous：同じ thread_id で前回保存した状態。初回は、空の状態から始める
    #  messages  ： 会話の履歴
    #  cart      ： 買い物かごの中身
    #  turn_count： エージェントが応答した回数
    state = previous or {"messages": [], "cart": [], "turn_count": 0}
    messages = add_messages(state["messages"], [HumanMessage(content=message)])
    cart = state["cart"]

    response = call_model(messages).result()
    # LLMがツール呼び出しを返す間は、ツールを実行して、もう一度LLMを呼ぶ
    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "add_to_cart":
                item = tool_call["args"]["item"]
                cart = add_to_cart.invoke({"item": item, "cart": cart})
                content = f"{item}を追加しました。かご：{cart}"
            else:
                cart_now = show_cart.invoke({"cart": cart})
                content = f"かごの中身：{cart_now}" if cart_now else "かごは空です。"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))

        messages = add_messages(messages, [response] + tool_messages)
        response = call_model(messages).result()
    messages = add_messages(messages, [response])

    # 戻り値が、次回の previous として保存される（応答した回数も、ここで数える）
    return {"messages": messages, "cart": cart, "turn_count": state["turn_count"] + 1}


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
    state = agent.invoke(message, config_a)
    print(f"[thread-A] AI：{state['messages'][-1].text}")

    # 会話Aの状態のうち、独自の項目（cart・turn_count）を表示する。
    print(f"[thread-A] 状態：cart={state['cart']}, turn_count={state['turn_count']}")


    # 2回目：同じ会話Aの続きで、牛乳をかごに入れる
    message = "牛乳も追加してください。"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    state = agent.invoke(message, config_a)
    print(f"[thread-A] AI：{state['messages'][-1].text}")

    # 会話Aの状態のうち、独自の項目（cart・turn_count）を表示する。
    print(f"[thread-A] 状態：cart={state['cart']}, turn_count={state['turn_count']}")


    # 3回目：同じ会話Aの続きで、かごの中身を確認する
    message = "今、かごに何が入っていますか？"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、config_aで、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    state = agent.invoke(message, config_a)
    print(f"[thread-A] AI：{state['messages'][-1].text}")

    # 会話Aの状態のうち、独自の項目（cart・turn_count）を表示する。
    print(f"[thread-A] 状態：cart={state['cart']}, turn_count={state['turn_count']}")

    # ##################################################################
    # 会話B：会話A とは異なる別の会話で、かごの中身を確認する
    # ##################################################################
    message = "かごの中身を見せてください。"
    print(f"[thread-B] 利用者：{message}")

    # エージェントを実行する。このとき、config_bで、会話Bを示すthread_id = thread-B を渡す。
    state = agent.invoke(message, config_b)
    print(f"[thread-B] AI：{state['messages'][-1].text}")

    # 会話Bの状態のうち、独自の項目（cart・turn_count）を表示する。会話Aとは別の状態なので、かごは空。
    print(f"[thread-B] 状態：cart={state['cart']}, turn_count={state['turn_count']}")


    # 状態は、エージェントの戻り値として、checkpointer に会話ごとに保存される。

if __name__ == "__main__":
    main()
