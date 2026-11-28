import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
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


# 予備のLLM：
#  通常のLLMが失敗し続けたときに代わりに利用
#  make_llm と同じ設定で、モデル名だけを変える
def make_fallback_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model="gpt-5-mini", reasoning_effort="low", use_responses_api=True, timeout=60, max_retries=1)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-sonnet-5-5", reasoning_effort="low", max_tokens=16000, timeout=60, max_retries=1)


llm = make_llm("low")

#####################################
# 架空の所在地データ
#####################################
# 住所の詳細データ（登録されている住所は1件と仮定する）
PRECISE_LOCATIONS = {
    "サンプル市中央1-2-3": "サンプルビルA（3階建ての事務所ビル）",
}
# 住所の広域データ（登録されている住所は1件と仮定する）
GENERAL_AREAS = {
    "サンプル市": "教材用の架空の市。駅を中心に商業地が広がっている。",
}

# 一時的な障害の再現用：ツール2は、奇数回目の呼び出しで通信エラーを起こす（すぐにやり直せば成功する）
flaky_state = {"calls": 0}


###############################################################################
# ツール1 : 住所の詳細情報を検索する。見つからなければ、例外（エラー）を発生させる
###############################################################################
@tool
def get_precise_location(address: str) -> str:
    """住所の詳細情報を検索する。

    Args:
        address: 番地まで含む住所。例：サンプル市中央1-2-3
    """
    if address not in PRECISE_LOCATIONS:
        raise LookupError(f"住所の詳細情報が見つかりません：{address}")
    return f"{address}：{PRECISE_LOCATIONS[address]}"


###############################################################################
# ツール2 : 住所の広域情報を検索する。詳細情報の検索に失敗したときに代わりに利用する
###############################################################################
# 03_05_Recovery.py との違い：奇数回目の呼び出しで、一時的な通信エラー（TimeoutError）を起こす。
# 同じ呼び出しをやり直せば成功するため、ToolRetryMiddleware が自動でやり直す様子を確かめられる
@tool
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
# 標準ミドルウェア1：ツールの一時的なエラーを、自動でやり直す
#####################################
# retry_on   : やり直す例外の種類。一時的な障害（TimeoutError）だけをやり直す。
#              LookupError（データが無い）は、何度やり直しても結果が同じなので、やり直さない
# on_failure : やり直しても失敗したとき。"error" は例外をそのまま外側（標準ミドルウェア2）へ渡す
tool_retry = ToolRetryMiddleware(
    max_retries=2,
    retry_on=(TimeoutError,),
    on_failure="error",
    initial_delay=0.5,
)


#####################################
# 標準ミドルウェア2：ツールのエラーを受け止めて、AIに伝える
#####################################
# 03_05_Recovery.py の handle_tool_errors（@wrap_tool_call で自作）に当たる。
# on_error が文字列を返すと、その文字列が「失敗した」という結果としてAIに渡る。
# None を返すと、例外はそのまま発生し、プログラムが止まる（想定外のエラーを隠さないため）
def on_tool_error(error: Exception, request: ToolCallRequest) -> str | None:
    if isinstance(error, LookupError):
        print(f"  [ツールのエラー] {error}")
        return f"失敗しました：{error}"
    return None


tool_error = ToolErrorMiddleware(on_error=on_tool_error)


#####################################
# 標準ミドルウェア3・4：LLMの呼び出しの失敗に備える（このサンプルでは、通常は発生しない）
#####################################
# ModelRetryMiddleware    : LLMの呼び出しが失敗したら、待ち時間を延ばしながらやり直す
# ModelFallbackMiddleware : やり直しても失敗したら、予備のLLMに切り替える
model_retry = ModelRetryMiddleware(max_retries=2)
model_fallback = ModelFallbackMiddleware(make_fallback_llm())


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[get_precise_location, get_general_area],
    # middleware    : 先に書いたものほど外側で動く。
    #                 LLM  ：やり直し（内側）→ それでも失敗したら予備のLLM（外側）
    #                 ツール：やり直し（内側）→ それでも失敗したらAIに伝える（外側）
    middleware=[model_fallback, model_retry, tool_error, tool_retry],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは所在地を案内する係です。次の手順で調べてください。\n"
        "1. まず get_precise_location で住所の詳細を調べる。\n"
        "2. 失敗した場合だけ、同じ検索を繰り返さず、get_general_area で市の情報を調べる。\n"
        "3. 市の情報しか分からなかった場合は、住所の詳細は分からなかったことを伝える。\n"
        "4. どちらも失敗した場合は、情報が見つからなかったことを伝え、推測で補わない。\n"
        "架空の教材用データであることを添えて、日本語で簡潔に答えてください。"
    ),
)


def show_tool_calls(messages) -> None:
    """エージェントが呼んだツールを表示する。"""
    for message in messages:
        if message.type == "ai":
            for call in message.tool_calls:
                print(f"  [ツール呼び出し] {call['name']}({call['args']})")


def main():
    """
    題材：ツールが失敗しても止まらず、代わりの方法に切り替えて答える。03_05_Recovery.py と同じ依頼を使う。
         違いは、エラーへの対処を標準ミドルウェアで組み立てていることと、
         ツール2で起きる一時的な通信エラーを、AIに伝える前に自動でやり直していること。
          1件目：住所の詳細が見つかる
          2件目：住所の詳細は見つからないが、市の情報は見つかる（一時的なエラーをやり直してから成功する）
          3件目：住所の詳細も市の情報も見つからない（やり直した後、データが無いことをAIに伝える）
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
        # エージェントを実行する（"user"：アプリを使う利用者の発言）
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        show_tool_calls(result["messages"])
        print(f"回答：{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
