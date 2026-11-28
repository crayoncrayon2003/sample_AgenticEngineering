import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
EVAL_DIR = os.path.join(ROOT, "outputs", "evaluations")


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
# テスト用の売上データ（架空）
#####################################
SALES_DATA = {
    "sales-A": [10, 20, 5],
    "sales-B": [-5, 10, 0],
    "sales-C": [],
}


###############################################################################
# ツール1 : 売上データを取得する
###############################################################################
@tool
def fetch_sales(dataset_id: str) -> list[int]:
    """指定した売上データを取得する。

    Args:
        dataset_id: 売上データのID。例：sales-A
    """
    return SALES_DATA[dataset_id]


###############################################################################
# ツール2 : 売上データの合計と行数を計算する
###############################################################################
@tool
def aggregate_sales(amounts: list[int]) -> dict:
    """売上データの合計と行数を計算する。空のデータは、合計も行数も0。

    Args:
        amounts: fetch_sales で取得した売上データ
    """
    return {"total": sum(amounts), "rows": len(amounts)}


#####################################
# LLMのレスポンス形式を定義する（最終回答）
#####################################
class SalesAnswer(BaseModel):
    #  dataset_id： 対象の売上データのID
    #  total     ： 合計
    #  rows      ： 行数
    dataset_id: str
    total: int
    rows: int


#####################################
# エージェント（評価の対象）
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[fetch_sales, aggregate_sales],
    # response_format: 最終回答の形（SalesAnswer）
    response_format=SalesAnswer,
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは売上を集計する係です。依頼されたデータを fetch_sales で取得し、"
        "取得した配列をそのまま aggregate_sales に渡してください。各ツールは1回ずつ使ってください。"
        "最終回答には、対象のIDと、ツールで得た合計・行数を入れてください。"
    ),
)


#####################################
# 評価：期待した結果と、実際の結果を比べる
#####################################
def evaluate(dataset_id: str, answer: dict, tool_names: list[str]) -> dict:
    amounts = SALES_DATA[dataset_id]
    expected = {"dataset_id": dataset_id, "total": sum(amounts), "rows": len(amounts)}
    checks = {
        "回答の数値とIDが正しい": answer == expected,
        "必要なツールを、正しい順番で1回ずつ使った": tool_names == ["fetch_sales", "aggregate_sales"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    """
    題材：テストケースでエージェントを評価する。回答・ツールの使い方・時間・トークン数を記録し、ファイルに保存する。
         テストケースは、合計3件ある（売上データ sales-A・sales-B・sales-C）。
    """

    records = []
    for dataset_id in SALES_DATA:
        print("----------------------------------------")
        print(f"テスト：{dataset_id}")

        # エージェントを実行し、時間を測る
        start = time.perf_counter()
        result = agent.invoke({"messages": [{"role": "user", "content": f"{dataset_id} の売上合計と行数を調べてください。"}]})
        elapsed = time.perf_counter() - start

        # 結果から、最終回答・使ったツール・トークン数を取り出す
        answer = result["structured_response"].model_dump()
        tool_names = [call["name"] for m in result["messages"] if m.type == "ai" for call in m.tool_calls]
        tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in result["messages"] if m.type == "ai")

        # 評価する
        evaluation = evaluate(dataset_id, answer, tool_names)
        print(f"回答：{answer}")
        print(f"使ったツール：{tool_names}")
        print(f"評価：{'合格' if evaluation['passed'] else '不合格'} {evaluation['checks']}")
        print(f"時間：{elapsed:.1f}秒、トークン数：{tokens}")

        records.append({
            "dataset_id": dataset_id,
            "answer": answer,
            "tool_names": tool_names,
            "evaluation": evaluation,
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": tokens,
        })

    # 全体の結果をまとめて、ファイルに保存する
    passed = sum(record["evaluation"]["passed"] for record in records)
    print("----------------------------------------")
    print(f"合格：{passed} / {len(records)}件")

    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"test_cases_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"passed": passed, "total": len(records), "records": records}, file, ensure_ascii=False, indent=2)
    print(f"保存先：{path}")


if __name__ == "__main__":
    main()
