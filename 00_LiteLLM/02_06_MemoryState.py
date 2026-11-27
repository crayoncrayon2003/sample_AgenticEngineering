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


llm = make_llm("low")  # minimalだと、ツールを呼ばずに「追加しました」と答えることがある


#####################################
# エージェントの状態（会話ごとに保存する）
#####################################
# thread_id ごとに、次の項目を持つ辞書を保存する
#  messages  ： 会話の履歴
#  cart      ： 買い物かごの中身
#  turn_count： エージェントが応答した回数
states = {}


def get_state(thread_id: str) -> dict:
    return states.get(thread_id, {"messages": [], "cart": [], "turn_count": 0})


#####################################
# ツール：状態を読む・更新する
#####################################
def add_to_cart(state: dict, item: str) -> str:
    """商品を買い物かごに追加する。"""
    state["cart"] = state["cart"] + [item]
    return f"{item}を追加しました。かご：{state['cart']}"


def show_cart(state: dict) -> str:
    """買い物かごの中身を返す。"""
    return f"かごの中身：{state['cart']}" if state["cart"] else "かごは空です。"


# LLMに渡すツールの説明（state は、プログラムから渡すので説明に含めない）
tools = [
    {"type": "function", "function": {
        "name": "add_to_cart",
        "description": "商品を買い物かごに追加する。",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string", "description": "追加する商品名"}},
            "required": ["item"],
        },
    }},
    {"type": "function", "function": {
        "name": "show_cart",
        "description": "買い物かごの中身を返す。",
        "parameters": {"type": "object", "properties": {}},
    }},
]


#####################################
# エージェント
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは買い物かごの係です。\n"
    "- 商品の追加を頼まれたら、商品ごとに必ずadd_to_cartを呼ぶ。\n"
    "- かごの中身を聞かれたら、必ずshow_cartを呼ぶ。\n"
    "- ツールを呼ばずに「追加しました」などと答えてはいけない。\n"
    "ツールの結果に基づいて、日本語で短く答えてください。"
)


def run_agent(message: str, thread_id: str) -> str:
    # 同じ thread_id で、前回保存した状態を読み出す（初回は空の状態）
    state = get_state(thread_id)
    state["messages"].append({"role": "user", "content": message})

    response = call_llm(llm, [{"role": "system", "content": system_prompt}] + state["messages"], tools=tools)
    # LLMがツール呼び出しを返す間は、ツールを実行して、もう一度LLMを呼ぶ
    while response.tool_calls:
        state["messages"].append(response.model_dump())
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "add_to_cart":
                result = add_to_cart(state, args["item"])
            else:
                result = show_cart(state)
            state["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, [{"role": "system", "content": system_prompt}] + state["messages"], tools=tools)
    state["messages"].append({"role": "assistant", "content": response.content})

    # 応答が終わるたびに回数を数え、状態を保存する（LangChain では、ミドルウェア after_agent で行っていた）
    state["turn_count"] += 1
    states[thread_id] = state
    return response.content


def main():
    """
    題材：買い物かごを管理するエージェント。かごの中身と、応答した回数を状態に保存する。
         会話A（thread-A）で3回、会話B（thread-B）で1回依頼する。
    """

    # thread_id：どの会話の続きかを表すID。会話ごとに別のIDを使う
    thread_a = "thread-A"
    thread_b = "thread-B"

    # ##################################################################
    # 会話A：1回目にりんご、2回目に牛乳をかごに入れて、3回目にかごの中身を確認する
    # ##################################################################
    # 1回目：りんごをかごに入れる
    message = "りんごをかごに入れてください。"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、会話Aを示すthread_id = thread-A を渡す。
    answer = run_agent(message, thread_a)
    print(f"[thread-A] AI：{answer}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = get_state(thread_a)
    print(f"[thread-A] 状態：cart={state['cart']}, turn_count={state['turn_count']}")


    # 2回目：同じ会話Aの続きで、牛乳をかごに入れる
    message = "牛乳も追加してください。"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    answer = run_agent(message, thread_a)
    print(f"[thread-A] AI：{answer}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = get_state(thread_a)
    print(f"[thread-A] 状態：cart={state['cart']}, turn_count={state['turn_count']}")


    # 3回目：同じ会話Aの続きで、かごの中身を確認する
    message = "今、かごに何が入っていますか？"
    print(f"[thread-A] 利用者：{message}")

    # エージェントを実行する。このとき、会話Aを示すthread_id = thread-A を渡すと、前の状態（かごの中身）を引き継ぐ。
    answer = run_agent(message, thread_a)
    print(f"[thread-A] AI：{answer}")

    # 会話Aの状態を取り出して、独自の項目（cart・turn_count）を表示する。
    state = get_state(thread_a)
    print(f"[thread-A] 状態：cart={state['cart']}, turn_count={state['turn_count']}")

    # ##################################################################
    # 会話B：会話A とは異なる別の会話で、かごの中身を確認する
    # ##################################################################
    message = "かごの中身を見せてください。"
    print(f"[thread-B] 利用者：{message}")

    # エージェントを実行する。このとき、会話Bを示すthread_id = thread-B を渡す。
    answer = run_agent(message, thread_b)
    print(f"[thread-B] AI：{answer}")

    # 会話Bの状態を取り出して、独自の項目（cart・turn_count）を表示する。会話Aとは別の状態なので、かごは空。
    state = get_state(thread_b)
    print(f"[thread-B] 状態：cart={state['cart']}, turn_count={state['turn_count']}")


    # 状態の形は、「エージェントの状態」を参照。状態は会話（thread_id）ごとに保存される。

if __name__ == "__main__":
    main()
