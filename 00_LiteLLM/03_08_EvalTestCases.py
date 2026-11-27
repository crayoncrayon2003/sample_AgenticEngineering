import json
import os
import time
from datetime import datetime

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
EVAL_DIR = os.path.join(ROOT, "outputs", "evaluations")


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
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
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
        num_retries=1,
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """プロンプトの {名前} に値を入れる（LangChain の ChatPromptTemplate に当たる）。"""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# テスト用の売上データ（架空）
#####################################
SALES_DATA = {
    "sales-A": [10, 20, 5],
    "sales-B": [-5, 10, 0],
    "sales-C": [],
}


#####################################
# ツール1 : 売上データを取得する
#####################################
def fetch_sales(dataset_id: str) -> list[int]:
    """指定した売上データを取得する。

    Args:
        dataset_id: 売上データのID。例：sales-A
    """
    return SALES_DATA[dataset_id]


#####################################
# ツール2 : 売上データの合計と行数を計算する
#####################################
def aggregate_sales(amounts: list[int]) -> dict:
    """売上データの合計と行数を計算する。空のデータは、合計も行数も0。

    Args:
        amounts: fetch_sales で取得した売上データ
    """
    return {"total": sum(amounts), "rows": len(amounts)}


#####################################
# LLMに渡すツールの説明
#####################################
# LangChain の @tool は、関数の docstring と型ヒントからこの説明を自動で作る。ここでは辞書で書く
tools = [
    {"type": "function", "function": {
        "name": "fetch_sales",
        "description": "指定した売上データを取得する。",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string", "description": "売上データのID。例：sales-A"}},
            "required": ["dataset_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "aggregate_sales",
        "description": "売上データの合計と行数を計算する。空のデータは、合計も行数も0。",
        "parameters": {
            "type": "object",
            "properties": {"amounts": {"type": "array", "items": {"type": "integer"}, "description": "fetch_sales で取得した売上データ"}},
            "required": ["amounts"],
        },
    }},
]
tool_functions = {"fetch_sales": fetch_sales, "aggregate_sales": aggregate_sales}


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
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは売上を集計する係です。依頼されたデータを fetch_sales で取得し、"
    "取得した配列をそのまま aggregate_sales に渡してください。各ツールは1回ずつ使ってください。"
    "最終回答には、対象のIDと、ツールで得た合計・行数を入れてください。"
)


# エージェントを実行し、最終回答・使ったツール・トークン数を返す
def run_agent(message: str) -> tuple[dict, list[str], int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    tool_names = []
    tokens = 0
    while True:
        # tools と response_format を両方渡す。ツールが不要になったら、SalesAnswer の形で答えが返る
        response = litellm.completion(
            model=llm["model"], reasoning_effort=llm["reasoning_effort"], max_tokens=llm["max_tokens"],
            messages=messages, tools=tools, response_format=SalesAnswer, timeout=60, num_retries=1,
        )
        tokens += response.usage.total_tokens
        message_ai = response.choices[0].message
        if not message_ai.tool_calls:
            return SalesAnswer.model_validate_json(message_ai.content).model_dump(), tool_names, tokens
        messages.append(message_ai)
        for tool_call in message_ai.tool_calls:
            tool_names.append(tool_call.function.name)
            result = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})


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
        answer, tool_names, tokens = run_agent(f"{dataset_id} の売上合計と行数を調べてください。")
        elapsed = time.perf_counter() - start

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
