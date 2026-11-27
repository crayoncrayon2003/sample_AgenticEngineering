import os
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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


llm = make_llm("minimal")

#####################################
# 1段目：タスク
#####################################
# 1つのタスクを生成する関数
def make_chain_task(instruction: str):
    prompt_task = ChatPromptTemplate.from_messages([
        # system : アプリを作った開発者からの指示
        ("system", instruction + "日本語で簡潔に出力してください。"),
        # user   : アプリを使う利用者の発言
        ("user", "テーマ：{topic}"),
    ])
    chain_task = prompt_task | llm | StrOutputParser()
    return chain_task

# 3つのタスク「要約」「よくある誤解」「用語」を作る
chain_summary        = make_chain_task("テーマを3文で要約してください。")
chain_misconceptions = make_chain_task("テーマについて、よくある誤解を2つ挙げてください。")
chain_terms          = make_chain_task("テーマの重要な用語を5つ、1行ずつ短い説明付きで挙げてください。")

#####################################
# 2段目：結果をまとめる処理
#####################################
prompt_merge = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "渡された要約・誤解・用語を使い、初学者向けの解説文を日本語で書いてください。"
        "誤解は正しい説明と対にしてください。400字程度にしてください。",
    ),
    (
        # user   : アプリを使う利用者の発言
        "user",
        "テーマ：{topic}\n\n【要約】\n{summary}\n\n"
        "【よくある誤解】\n{misconceptions}\n\n【用語】\n{terms}",
    ),
])
chain_merge = prompt_merge | llm | StrOutputParser()


#####################################
# グラフの状態を定義する
#####################################
class ParallelState(TypedDict, total=False):
    #  topic         ： テーマ
    #  summary       ： 要約
    #  misconceptions： よくある誤解
    #  terms         ： 用語
    #  answer        ： 統合した解説文
    topic: str
    summary: str
    misconceptions: str
    terms: str
    answer: str


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# 1段目：3つのノードは、それぞれ別の項目を更新する
def node_summary(state: ParallelState) -> dict:
    return {"summary": chain_summary.invoke({"topic": state["topic"]})}


def node_misconceptions(state: ParallelState) -> dict:
    return {"misconceptions": chain_misconceptions.invoke({"topic": state["topic"]})}


def node_terms(state: ParallelState) -> dict:
    return {"terms": chain_terms.invoke({"topic": state["topic"]})}


# 2段目：3つの結果をまとめる
def node_merge(state: ParallelState) -> dict:
    answer = chain_merge.invoke({
        "topic": state["topic"],
        "summary": state["summary"],
        "misconceptions": state["misconceptions"],
        "terms": state["terms"],
    })
    return {"answer": answer}


#####################################
# グラフを組み立てる
#####################################
#  START ─┬→ summary        ─┐
#         ├→ misconceptions ─┼→ merge → END
#         └→ terms          ─┘
# START から3つのノードへ辺を引くと、3つが同時に実行される。
# merge は、3つすべてが終わってから実行される。
graph = StateGraph(ParallelState)
graph.add_node("summary", node_summary)
graph.add_node("misconceptions", node_misconceptions)
graph.add_node("terms", node_terms)
graph.add_node("merge", node_merge)
graph.add_edge(START, "summary")
graph.add_edge(START, "misconceptions")
graph.add_edge(START, "terms")
graph.add_edge("summary", "merge")
graph.add_edge("misconceptions", "merge")
graph.add_edge("terms", "merge")
graph.add_edge("merge", END)
app = graph.compile()


def main():
    """
    題材：あるテーマについて「要約」「よくある誤解」「用語」を並列実行する。実行結果を解説文に統合する。
    """

    topic = "AIエージェント"
    print(f"テーマ：{topic}")

    # 1段目（3つを並列実行）と2段目（統合）を、まとめて実行する
    start = time.perf_counter()
    result = app.invoke({"topic": topic})
    print(f"--- 全体の実行時間：{time.perf_counter() - start:.1f}秒 ---")

    print("[summary]")
    print(result["summary"])
    print("[misconceptions]")
    print(result["misconceptions"])
    print("[terms]")
    print(result["terms"])
    print("--- 2段目：統合した解説文 ---")
    print(result["answer"])

    # 比較のため、同じ処理（1段目の3つと、2段目の統合）を、すべて順番に実行する
    start = time.perf_counter()
    summary = chain_summary.invoke({"topic": topic})
    misconceptions = chain_misconceptions.invoke({"topic": topic})
    terms = chain_terms.invoke({"topic": topic})
    chain_merge.invoke({"topic": topic, "summary": summary, "misconceptions": misconceptions, "terms": terms})
    print(f"（参考）すべて順番に実行した場合：{time.perf_counter() - start:.1f}秒")


if __name__ == "__main__":
    main()
