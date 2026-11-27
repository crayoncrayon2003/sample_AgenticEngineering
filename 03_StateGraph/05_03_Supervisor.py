import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from typing import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph

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
# 窓口担当：どの専門担当に任せるかを決めるチェーン
#####################################
# LLMのレスポンス形式を定義する（窓口担当の判断）
class Assignment(BaseModel):
    #  reason       ： 判断の理由
    #  need_booking ： 予約担当に任せるなら True
    #  need_info    ： 情報担当に任せるなら True
    #  理由 → 判断 の順に並べ、理由を書き出してから判断させる
    reason: str = Field(description="判断の理由（日本語で1文）")
    need_booking: bool = Field(description="航空券やホテルの予約が含まれていれば true（日付などの詳細が足りなくても true）")
    need_info: bool = Field(description="一般的な質問（観光名所など）が含まれていれば true")


prompt_assign = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは利用者の依頼を受け付ける窓口担当です。依頼に、予約と質問のどちらが含まれるかを判断してください。"
        "両方含まれていれば、両方を true にしてください。意図がはっきりしなければ、両方を false にしてください。"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "{request}"
    ),
])
chain_assign = prompt_assign | llm.with_structured_output(Assignment, method="json_schema")

#####################################
# 窓口担当：専門担当の結果をまとめて答えるチェーン
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは利用者の依頼を受け付ける窓口担当です。専門担当の結果を1つの回答にまとめて、日本語で簡潔に伝えてください。"
        "専門担当の結果が無い場合は、何に困っているのかを具体的に尋ねてください。"
        "専門担当の結果に無い内容（日程・候補・料金など）を付け加えず、模擬処理を実際に完了したように説明しないでください。"
    ),(
        # user   : 依頼と、専門担当の結果
        "user",
        "【依頼】\n{request}\n\n【予約担当の結果】\n{booking}\n\n【情報担当の結果】\n{info}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


# 専門担当のエージェントに依頼し、最後の回答だけを返す
def ask(agent, request: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


#####################################
# グラフの状態を定義する
#####################################
class DeskState(TypedDict, total=False):
    #  request     ： 利用者の依頼
    #  need_booking： 予約担当に任せるか
    #  need_info   ： 情報担当に任せるか
    #  booking     ： 予約担当の結果
    #  info        ： 情報担当の結果
    #  answer      ： 利用者への回答
    request: str
    need_booking: bool
    need_info: bool
    booking: str
    info: str
    answer: str


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# 1段目：窓口担当が、どの専門担当に任せるかを決める
def node_assign(state: DeskState) -> dict:
    assignment = chain_assign.invoke({"request": state["request"]})
    print(f"  [窓口担当の判断] 予約担当：{assignment.need_booking}、情報担当：{assignment.need_info}（{assignment.reason}）")
    return {"need_booking": assignment.need_booking, "need_info": assignment.need_info,
            "booking": "（任せていない）", "info": "（任せていない）"}


# 2段目：専門担当（予約担当・情報担当）
def node_booking(state: DeskState) -> dict:
    print("  → [窓口担当] 予約担当に任せます。")
    return {"booking": ask(agent_booking, state["request"])}


def node_info(state: DeskState) -> dict:
    print("  → [窓口担当] 情報担当に任せます。")
    return {"info": ask(agent_info, state["request"])}


# 3段目：窓口担当が、専門担当の結果をまとめて答える
def node_answer(state: DeskState) -> dict:
    return {"answer": chain_answer.invoke({"request": state["request"], "booking": state["booking"], "info": state["info"]})}


# 条件分岐：任せる専門担当を返す。2人とも返すと、2つのノードが同時に実行される
def select_specialists(state: DeskState) -> list[str]:
    specialists = []
    if state["need_booking"]:
        specialists.append("booking")
    if state["need_info"]:
        specialists.append("info")
    if not specialists:
        # どちらにも任せないときは、そのまままとめ（利用者に内容を尋ねる）へ進む
        return ["answer"]
    return specialists


#####################################
# グラフを組み立てる
#####################################
#  START → assign（窓口担当の判断）─┬→ booking（予約担当）─┐
#                                  ├→ info（情報担当）   ─┼→ answer（窓口担当がまとめる）→ END
#                                  └───────────────────────┘（どちらにも任せない）
graph = StateGraph(DeskState)
graph.add_node("assign", node_assign)
graph.add_node("booking", node_booking)
graph.add_node("info", node_info)
graph.add_node("answer", node_answer)
graph.add_edge(START, "assign")
graph.add_conditional_edges("assign", select_specialists, ["booking", "info", "answer"])
graph.add_edge("booking", "answer")
graph.add_edge("info", "answer")
graph.add_edge("answer", END)
app = graph.compile()


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
        # 窓口担当の判断 → 専門担当 → 窓口担当がまとめる、を実行する
        result = app.invoke({"request": request})
        print(f"最終回答：{result['answer']}")


if __name__ == "__main__":
    main()
