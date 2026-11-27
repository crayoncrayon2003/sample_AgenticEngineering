import json
import os
import time

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
            "fallbacks": ["openai/gpt-5-mini"],  # 予備のモデル
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "fallbacks": ["anthropic/claude-sonnet-5-5"],  # 予備のモデル
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
        # num_retries：LLMの呼び出しが失敗したら、やり直す回数（LangChain の ModelRetryMiddleware に当たる）
        num_retries=2,
        # fallbacks  ：やり直しても失敗したら、順に試す予備のモデル（LangChain の ModelFallbackMiddleware に当たる）
        fallbacks=llm["fallbacks"],
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """プロンプトの {名前} に値を入れる（LangChain の ChatPromptTemplate に当たる）。"""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# 架空の所在地データ
#####################################
PRECISE_LOCATIONS = {
    "サンプル市中央1-2-3": "サンプルビルA（3階建ての事務所ビル）",
}
GENERAL_AREAS = {
    "サンプル市": "教材用の架空の市。駅を中心に商業地が広がっている。",
}

# 一時的な障害の再現用：ツール2は、奇数回目の呼び出しで通信エラーを起こす（すぐにやり直せば成功する）
flaky_state = {"calls": 0}


#####################################
# ツール1 : 住所の詳細情報を検索する。見つからなければ、例外（エラー）を発生させる
#####################################
def get_precise_location(address: str) -> str:
    """住所の詳細情報を検索する。

    Args:
        address: 番地まで含む住所。例：サンプル市中央1-2-3
    """
    if address not in PRECISE_LOCATIONS:
        raise LookupError(f"住所の詳細情報が見つかりません：{address}")
    return f"{address}：{PRECISE_LOCATIONS[address]}"


#####################################
# ツール2 : 住所の広域情報を検索する。詳細情報の検索に失敗したときに代わりに利用する
#####################################
def get_general_area(city: str) -> str:
    """住所の広域情報を検索する。詳細情報の検索に失敗したときに代わりに利用する

    Args:
        city: 市の名前。例：サンプル市
    """
    flaky_state["calls"] += 1
    if flaky_state["calls"] % 2 == 1:
        print("  [一時的なエラー] 通信がタイムアウトしました（やり直します）")
        raise TimeoutError("通信がタイムアウトしました。")
    if city not in GENERAL_AREAS:
        raise LookupError(f"市の情報も見つかりません：{city}")
    return f"{city}：{GENERAL_AREAS[city]}"


#####################################
# LLMに渡すツールの説明
#####################################
# LangChain の @tool は、関数の docstring と型ヒントからこの説明を自動で作る。ここでは辞書で書く
tools = [
    {"type": "function", "function": {
        "name": "get_precise_location",
        "description": "住所の詳細情報を検索する。",
        "parameters": {
            "type": "object",
            "properties": {"address": {"type": "string", "description": "番地まで含む住所。例：サンプル市中央1-2-3"}},
            "required": ["address"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_general_area",
        "description": "住所の広域情報を検索する。詳細情報の検索に失敗したときに代わりに利用する",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "市の名前。例：サンプル市"}},
            "required": ["city"],
        },
    }},
]
tool_functions = {"get_precise_location": get_precise_location, "get_general_area": get_general_area}


#####################################
# 一時的なツールエラーを、自動リトライする（LangChain の ToolRetryMiddleware に当たる）
#####################################
# LiteLLM が、引き受けるのはLLMの呼び出しまでである。ツールの実行は、自分のコードで行う。
# そのため、ツールの自動リトライは自分で書く。
# リトライは、次の条件実行する
# 　* リトライする　：　一時的な障害（TimeoutError）
# 　* リトライしない：　その他。例えば、何度やり直しても結果が同じになるケースなど。具体的には、データが無い（LookupError）など。
TOOL_MAX_RETRIES = 2


def call_with_retry(name: str, args: dict) -> str:
    for attempt in range(TOOL_MAX_RETRIES + 1):
        try:
            return tool_functions[name](**args)
        except TimeoutError:
            if attempt == TOOL_MAX_RETRIES:
                raise
            time.sleep(0.5 * (2 ** attempt))  # 待ち時間を延ばしながらやり直す


#####################################
# ツールのエラーを受け止めて、AIに伝える（LangChain の ToolErrorMiddleware に当たる）
#####################################
# ツールで例外が起きても、プログラムを止めずに「失敗した」という結果としてAIに返す。
# AIはその結果を見て、代わりの方法（ツール2）に切り替えられる
def run_tool(name: str, args: dict) -> str:
    print(f"  [ツール呼び出し] {name}({args})")
    try:
        return call_with_retry(name, args)
    except LookupError as error:
        print(f"  [ツールのエラー] {error}")
        return f"失敗しました：{error}"


#####################################
# エージェント
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは所在地を案内する係です。次の手順で調べてください。\n"
    "1. まず get_precise_location で住所の詳細を調べる。\n"
    "2. 失敗した場合だけ、同じ検索を繰り返さず、get_general_area で市の情報を調べる。\n"
    "3. 市の情報しか分からなかった場合は、住所の詳細は分からなかったことを伝える。\n"
    "4. どちらも失敗した場合は、情報が見つからなかったことを伝え、推測で補わない。\n"
    "架空の教材用データであることを添えて、日本語で簡潔に答えてください。"
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
    題材：ツールが失敗しても止まらず、代わりの方法に切り替えて答える。
         依頼は、合計3件ある。
          1件目：住所の詳細が見つかる
          2件目：住所の詳細は見つからないが、市の情報は見つかる（一時的なエラーをやり直してから成功する）
          3件目：住所の詳細も市の情報も見つからない（やり直した後、データが無いことを伝える）
         03_05_Recovery.py との違いは、市の検索で起きる一時的な通信エラーを自動でやり直すことと、
         LLMの呼び出しの失敗に備えて、やり直しと予備のLLMへの切り替えを設定していること。
    """

    # 3件の依頼
    requests = [
        "サンプル市中央1-2-3 の所在地情報を調べてください。",
        "サンプル市中央9-9-9 の所在地情報を調べてください。",
        "未登録市中央9-9-9 の所在地情報を調べてください。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")
        # エージェントを実行する。ツールの呼び出しとエラーは、実行中に表示する
        messages = run_agent(request)
        print(f"回答：{messages[-1].content}")


if __name__ == "__main__":
    main()
