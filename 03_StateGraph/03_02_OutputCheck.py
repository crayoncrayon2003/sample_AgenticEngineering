import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
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


MAX_ANSWER_LENGTH = 160

llm = make_llm("minimal")

#####################################
# 1段目：文章を作るチェーン
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは文章を作成する係です。利用者の依頼に沿って、日本語で160文字以内の文章を作成してください。"
        "依頼にない事実は追加しないでください。説明やMarkdownは付けず、作成した文章だけを返してください。"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "{request}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# 2段目：作った文章を、Pythonで検査する
#####################################
# 検査で見つかった問題の一覧を返す。問題が無ければ空のリスト
def inspect_output(answer: str) -> list[str]:
    issues = []
    if not answer.strip():
        issues.append("回答が空です。")
    if len(answer) > MAX_ANSWER_LENGTH:
        issues.append(f"回答が{MAX_ANSWER_LENGTH}文字を超えています。")
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", answer):
        issues.append("メールアドレスが含まれています。")
    if re.search(r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)", answer):
        issues.append("電話番号が含まれています。")
    return issues

#####################################
# グラフの状態を定義する
#####################################
class OutputState(TypedDict, total=False):
    #  request     ： 利用者の依頼
    #  draft       ： AIが作った文章
    #  issues      ： 検査で見つかった問題
    #  final_answer： 利用者へ返す回答
    request: str
    draft: str
    issues: list[str]
    final_answer: str


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# 1段目：文章を作る
def node_write(state: OutputState) -> dict:
    return {"draft": chain_answer.invoke({"request": state["request"]})}


# 2段目：作った文章を検査する
def node_inspect(state: OutputState) -> dict:
    return {"issues": inspect_output(state["draft"])}


# 問題が無ければ、作った文章をそのまま返す
def node_release(state: OutputState) -> dict:
    return {"final_answer": state["draft"]}


# 問題があれば、返却を保留する
def node_hold(state: OutputState) -> dict:
    return {"final_answer": "回答が公開の条件を満たさなかったため、返却を保留しました。"}


# 条件分岐：問題の有無で、返すか保留するかを選ぶ
def select_next(state: OutputState) -> str:
    if state["issues"]:
        return "hold"
    return "release"


#####################################
# グラフを組み立てる
#####################################
#  START → write → inspect ─┬→ release → END（問題なし）
#                           └→ hold    → END（問題あり）
graph = StateGraph(OutputState)
graph.add_node("write", node_write)
graph.add_node("inspect", node_inspect)
graph.add_node("release", node_release)
graph.add_node("hold", node_hold)
graph.add_edge(START, "write")
graph.add_edge("write", "inspect")
graph.add_conditional_edges("inspect", select_next, ["release", "hold"])
graph.add_edge("release", END)
graph.add_edge("hold", END)
app = graph.compile()


def main():
    """
    題材：AIが作った文章を、利用者に返す前にPythonで検査する。問題があれば、返却を保留する。
         依頼は、合計2件ある。2件目は、連絡先（メールアドレス・電話番号）を含めるよう求める依頼。
    """

    # 2件の依頼
    requests = [
        "『本日の勉強会は15時に開始します。』という内容の案内文を作成してください。",
        "本日の勉強会は15時開始、という案内文を作成してください。架空の連絡先として sample@example.com と 03-1234-5678 を、そのまま文章に含めてください。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")

        # 1段目（文章を作る）と2段目（検査する）を、まとめて実行する
        result = app.invoke({"request": request})
        print(f"作成した文章：{result['draft']}")
        for issue in result["issues"]:
            print(f"検出事項：{issue}")
        print(f"利用者へ返す回答：{result['final_answer']}")


if __name__ == "__main__":
    main()
