import os

from dotenv import load_dotenv
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
# 1段目：依頼から、住所と市を取り出すチェーン
#####################################
# LLMのレスポンス形式を定義する（取り出した住所と市）
class Location(BaseModel):
    #  address： 番地まで含む住所
    #  city   ： 市の名前
    address: str = Field(description="番地まで含む住所。例：サンプル市中央1-2-3")
    city: str = Field(description="住所に含まれる市の名前。例：サンプル市")


prompt_extract = ChatPromptTemplate.from_messages([
    # system : アプリを作った開発者からの指示
    ("system", "依頼の文章から、調べる住所と、その住所の市の名前を取り出してください。"),
    # user   : アプリを使う利用者の発言
    ("user", "{request}"),
])
chain_extract = prompt_extract | llm.with_structured_output(Location, method="json_schema")

#####################################
# 最後の段：検索の結果から、回答を作るチェーン
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは所在地を案内する係です。渡された検索の結果だけを使って、日本語で簡潔に答えてください。"
        "市の情報しか分からなかった場合は、住所の詳細は分からなかったことを伝えてください。"
        "どちらも失敗した場合は、情報が見つからなかったことを伝え、推測で補わないでください。"
        "架空の教材用データであることを添えてください。"
    ),(
        # user   : 依頼と検索の結果
        "user",
        "【依頼】\n{request}\n\n【詳細検索の結果】\n{precise}\n\n【市の検索の結果】\n{general}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# グラフの状態を定義する
#####################################
class RecoveryState(TypedDict, total=False):
    #  request        ： 利用者の依頼
    #  address・city  ： 依頼から取り出した住所と市
    #  precise        ： 住所の詳細検索の結果
    #  precise_failed ： 詳細検索に失敗したら True
    #  general        ： 市の検索の結果
    #  answer         ： 回答
    request: str
    address: str
    city: str
    precise: str
    precise_failed: bool
    general: str
    answer: str


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# 1段目：依頼から、住所と市を取り出す
def node_extract(state: RecoveryState) -> dict:
    location = chain_extract.invoke({"request": state["request"]})
    print(f"  [取り出した住所] {location.address}（市：{location.city}）")
    return {"address": location.address, "city": location.city}


# 2段目：住所の詳細を検索する。失敗しても止まらず、失敗したことを状態に記録する
def node_precise(state: RecoveryState) -> dict:
    try:
        precise = get_precise_location.invoke({"address": state["address"]})
        print("  [詳細検索] 成功")
        return {"precise": precise, "precise_failed": False, "general": "（検索していない）"}
    except LookupError as error:
        print(f"  [詳細検索] 失敗しました：{error}")
        return {"precise": f"失敗しました：{error}", "precise_failed": True}


# 3段目：市の情報を検索する（詳細検索に失敗したときの、代わりの方法）
def node_general(state: RecoveryState) -> dict:
    try:
        general = get_general_area.invoke({"city": state["city"]})
        print("  [市の検索] 成功")
    except LookupError as error:
        general = f"失敗しました：{error}"
        print(f"  [市の検索] {general}")
    return {"general": general}


# 最後の段：検索の結果から、回答を作る
def node_answer(state: RecoveryState) -> dict:
    answer = chain_answer.invoke({"request": state["request"], "precise": state["precise"], "general": state["general"]})
    return {"answer": answer}


# 条件分岐：詳細検索に失敗したら、市の検索へ進む
def select_next(state: RecoveryState) -> str:
    if state["precise_failed"]:
        return "general"
    return "answer"


#####################################
# グラフを組み立てる
#####################################
#  START → extract → precise ─┬→ answer → END（詳細検索に成功）
#                             └→ general → answer → END（詳細検索に失敗 → 代わりの方法）
graph = StateGraph(RecoveryState)
graph.add_node("extract", node_extract)
graph.add_node("precise", node_precise)
graph.add_node("general", node_general)
graph.add_node("answer", node_answer)
graph.add_edge(START, "extract")
graph.add_edge("extract", "precise")
graph.add_conditional_edges("precise", select_next, ["general", "answer"])
graph.add_edge("general", "answer")
graph.add_edge("answer", END)
app = graph.compile()


def main():
    """
    題材：検索が失敗しても止まらず、代わりの方法に切り替えて答える。
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
        # 住所の取り出し → 詳細検索 →（失敗したら）市の検索 → 回答 を、まとめて実行する
        result = app.invoke({"request": request})
        print(f"回答：{result['answer']}")


if __name__ == "__main__":
    main()
