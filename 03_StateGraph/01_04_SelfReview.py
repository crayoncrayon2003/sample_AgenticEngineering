import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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


llm = make_llm("minimal")

#####################################
# LLMのレスポンス形式を定義する（評価結果）
#####################################
class Review(BaseModel):
    #  passed： 要件をすべて満たし、修正が不要なら True。
    #  issues： 修正すべき点の一覧。合格なら空のリスト。
    passed: bool      = Field(description="要件をすべて満たし、修正不要ならtrue")
    issues: list[str] = Field(description="修正すべき点。合格なら空のリスト")


#####################################
# 作成者（初回の作成と、指摘を受けた修正を兼ねる）
#####################################
prompt_write = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたはPythonプログラマーです。課題を満たすコードを書いてください。"
        "前回のコードと指摘がある場合は、指摘に対応した修正後のコード全体を書いてください。"
        "説明文やMarkdownのコードフェンスを付けず、Pythonコードだけを出力してください。"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "【課題】\n{task}\n\n【前回のコード】\n{code}\n\n【指摘】\n{issues}"
    ),
])
chain_write = prompt_write | llm | StrOutputParser()

#####################################
# 評価者
#####################################
prompt_review = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたはコードレビュー担当です。課題の要件、バグ、境界値、docstringを確認してください。"
        "コードは実行できないため、読んで判断してください。",
    ),
    (
        # user   : アプリを使う利用者の発言
        "user",
        "【課題】\n{task}\n\n【コード】\n{code}"),
])
chain_review = prompt_review | llm.with_structured_output(Review, method="json_schema")


#####################################
# グラフの状態を定義する
#####################################
class ReviewState(TypedDict, total=False):
    #  task      ： 課題
    #  code      ： その時点の最新版のコード（作成するたびに更新する）
    #  issues    ： 評価者からの指摘（評価するたびに更新する）
    #  passed    ： 評価が合格なら True
    #  round_no  ： 何回目の作成か
    #  max_rounds： 繰り返しの上限
    task: str
    code: str
    issues: str
    passed: bool
    round_no: int
    max_rounds: int


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# Step1 作成者：コードを作成する（2回目以降は、指摘に沿って修正する）
def node_write(state: ReviewState) -> dict:
    result_write = chain_write.invoke({"task": state["task"], "code": state["code"], "issues": state["issues"]})
    round_no = state["round_no"] + 1
    print(f"--- 第{round_no}版 ---")
    print(result_write)
    return {"code": result_write, "round_no": round_no}


# Step2 評価者：コードを評価する
def node_review(state: ReviewState) -> dict:
    result_review = chain_review.invoke({"task": state["task"], "code": state["code"]})
    if result_review.passed:
        print("評価：合格")
    else:
        print("評価：要修正")
        for issue in result_review.issues:
            print(f"  - {issue}")
    return {"passed": result_review.passed, "issues": "\n".join(result_review.issues)}


# 条件分岐：合格、または上限回数に達したら終了。そうでなければ、作成者に戻る
def select_next(state: ReviewState) -> str:
    if state["passed"]:
        return END
    if state["round_no"] >= state["max_rounds"]:
        print("上限回数に達したため、最後の版で終了します。")
        return END
    return "write"


#####################################
# グラフを組み立てる
#####################################
#  START → write → review ─┬→ END（合格、または上限回数）
#            ↑             │
#            └─────────────┘（要修正）
graph = StateGraph(ReviewState)
graph.add_node("write", node_write)
graph.add_node("review", node_review)
graph.add_edge(START, "write")
graph.add_edge("write", "review")
graph.add_conditional_edges("review", select_next, ["write", END])
app = graph.compile()


def main():
    """
    題材：Pythonの関数を書かせて、別のプロンプトで評価し、指摘に沿って直す。
         合格するか、上限回数（3回）に達するまで繰り返す。
    """

    # 課題（作成する関数の要件）
    task_text = (
        "format_duration という関数を作成してください。\n"
        "1. 秒数の整数 seconds を受け取り、「1日2時間3分4秒」の形の文字列を返す。\n"
        "2. 0になる単位は省略する（例：3600 → 「1時間」、90061 → 「1日1時間1分1秒」）。\n"
        "3. seconds が0なら「0秒」を返す。\n"
        "4. seconds が int でない場合は TypeError を発生させる。bool も int ではないものとして扱う。\n"
        "5. seconds が負なら ValueError を発生させる。\n"
        "6. docstring は日本語で書き、doctest形式の使用例を3つ以上含める。"
    )
    # 状態の初期値。初回は前回のコードも指摘も無いため、「（なし）」を渡す
    initial_state = {
        "task": task_text,
        "code": "（なし）",
        "issues": "（なし）",
        "round_no": 0,
        "max_rounds": 3,
    }
    # 作成 → 評価 を繰り返し、最終版のコードを受け取る
    result = app.invoke(initial_state)
    print("--- 最終コード ---")
    print(result["code"])


if __name__ == "__main__":
    main()
