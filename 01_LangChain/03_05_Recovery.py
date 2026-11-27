import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langchain.messages import ToolMessage
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
@tool
def get_general_area(city: str) -> str:
    """住所の広域情報を検索する。詳細情報の検索に失敗したときに代わりに利用する

    Args:
        city: 市の名前。例：サンプル市
    """
    if city not in GENERAL_AREAS:
        raise LookupError(f"市の情報も見つかりません：{city}")
    return f"{city}：{GENERAL_AREAS[city]}"


#####################################
# ミドルウェア：ツールのエラーを受け止めて、AIに伝える
#####################################
# ツールで例外が起きても、プログラムを止めずに「失敗した」という結果としてAIに返す。
# AIはその結果を見て、代わりの方法（ツール2）に切り替えられる
@wrap_tool_call
def handle_tool_errors(request: ToolCallRequest, handler):
    try:
        # handler：ツール本体を実行する
        return handler(request)
    except LookupError as error:
        print(f"  [ツールのエラー] {error}")
        return ToolMessage(
            content=f"失敗しました：{error}",
            tool_call_id=request.tool_call["id"],
            status="error",
        )


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[get_precise_location, get_general_area],
    # middleware    : ツールのエラーを受け止める処理
    middleware=[handle_tool_errors],
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
    題材：ツールが失敗しても止まらず、代わりの方法に切り替えて答える。
         依頼は、合計3件ある。
          1件目：住所の詳細が見つかる
          2件目：住所の詳細は見つからないが、市の情報は見つかる（代わりの方法に切り替える）
          3件目：住所の詳細も市の情報も見つからない
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
