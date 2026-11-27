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


llm = make_llm("low")


#####################################
# 専門担当1：予約担当
#####################################
# 予約担当が処理するタスク（模擬タスク）
@tool
def book_travel(request: str) -> str:
    """航空券やホテルの予約を処理する（模擬。実際には予約しない）。

    Args:
        request: 予約の依頼内容
    """
    print("      → [予約担当のツール] 予約処理を模擬しました。")
    return f"予約の依頼「{request}」を受け付けました（模擬処理のため、実際の予約はしていません）。"


agent_booking = create_agent(
    model=llm,
    tools=[book_travel],
    system_prompt=(
        "あなたは航空券とホテルの予約担当です。依頼を受けたら必ずbook_travelを使い、結果を日本語で簡潔に伝えてください。"
        "ツールの結果に無い内容（日程・候補・料金など）を付け加えないでください。"
    ),
)


#####################################
# 専門担当2：情報担当
#####################################
# 情報担当が調べられる情報（模擬データ）
INFO = {
    "パリの観光名所": "エッフェル塔、ルーブル美術館、凱旋門、モンマルトルの丘。",
    "世界で最も高い山": "エベレスト（標高8,848メートル）。",
}


@tool
def search_info(topic: Literal["パリの観光名所", "世界で最も高い山"]) -> str:
    """一般的な情報を検索する（模擬データ）。

    Args:
        topic: 調べたい話題（選択肢の中から選ぶ）
    """
    print(f"      → [情報担当のツール] 「{topic}」を検索しました。")
    return INFO.get(topic, f"「{topic}」の情報は見つかりませんでした。")


agent_info = create_agent(
    model=llm,
    tools=[search_info],
    system_prompt=(
        "あなたは一般的な質問の担当です。依頼を受けたら必ずsearch_infoで調べてください。"
        "検索結果だけを使って日本語で答え、見つからなければ、見つからなかったと伝えてください。"
    ),
)


#####################################
# 窓口担当が使うツール：専門担当に仕事を任せる
#####################################
@tool
def delegate_booking(request: str) -> str:
    """航空券やホテルの予約の依頼を、予約担当に任せる。

    Args:
        request: 予約の依頼内容
    """
    print("  → [窓口担当] 予約担当に任せます。")
    result = agent_booking.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def delegate_info(question: str) -> str:
    """一般的な質問を、情報担当に任せる。

    Args:
        question: 質問の内容
    """
    print("  → [窓口担当] 情報担当に任せます。")
    result = agent_info.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text


#####################################
# 窓口担当のエージェント
#####################################
agent_coordinator = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : 専門担当に仕事を任せるツール
    tools=[delegate_booking, delegate_info],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは利用者の依頼を受け付ける窓口担当です。依頼の内容に応じて、専門担当に仕事を任せてください。"
        "自分の知識で直接答えないでください。\n"
        "- 航空券やホテルの予約：delegate_booking（日付などの詳細が足りなくても、必ず任せる）\n"
        "- 一般的な質問：delegate_info\n"
        "- 1つの依頼に、予約と質問の両方が含まれる場合：両方の専門担当に任せる\n"
        "- 意図がはっきりしない依頼：ツールを使わず、利用者に具体的な内容を尋ねる\n"
        "専門担当の結果を受け取ったら、1つの回答にまとめて、日本語で簡潔に伝えてください。"
        "専門担当の結果に無い内容（日程・候補・料金など）を付け加えないでください。"
        "模擬処理を、実際に完了したように説明しないでください。"
    ),
)


def main():
    """
    題材：
        窓口担当のエージェントが利用者の依頼を受け付け、内容に応じて、専門担当のエージェントに仕事を任せる。
        1つの依頼に複数の用件があれば、複数の専門担当に任せ、結果を1つの回答にまとめる。

    登場人物：
        窓口担当：利用者の依頼を受け付ける。どの専門担当に任せるかを自分で判断し、結果をまとめて答える。
        予約担当：専門担当の1人。航空券やホテルの予約を処理する。
        情報担当：専門担当の1人。観光名所などの一般的な情報を調べる。

    前提：
        依頼は、合計4件ある。
          1. 予約だけの依頼          → 窓口担当は、予約担当に任せる。
          2. 質問だけの依頼          → 窓口担当は、情報担当に任せる。
          3. 予約と質問の両方の依頼  → 窓口担当は、2人の専門担当に任せ、2つの結果を1つの回答にまとめる。
          4. 内容がはっきりしない依頼 → 窓口担当は、専門担当に任せず、利用者に内容を尋ねる。

    流れ（1件分）：
        1. 窓口担当が、依頼の内容を読み、どの専門担当に任せるかを決める。
        2. 任された専門担当が、自分のツールを使って仕事をし、結果を窓口担当に返す。
        3. 窓口担当が、専門担当の結果をまとめて、利用者に答える。
    """

    # 4件の依頼
    requests = [
        "パリのホテルを予約してください。",
        "パリの観光名所を教えてください。",
        "パリのホテルを予約して、パリの観光名所も教えてください。",
        "例の件、お願いします。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")
        # 窓口担当のエージェントを実行する
        result = agent_coordinator.invoke({"messages": [{"role": "user", "content": request}]})
        print(f"最終回答：{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
