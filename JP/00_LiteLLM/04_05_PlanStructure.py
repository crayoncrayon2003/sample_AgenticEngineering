import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total.json")


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
# 操作（計画の部品）：前提条件を満たさないと ValueError を発生させる
#####################################
#  inspect  ： データの内容と、整数に変換できない値（不正値）の数を確認する
#  fill_zero   ： 不正値を0に置き換える（inspect が前提）。データの件数は変わらない
#  drop_invalid： 不正値のデータを取り除く（inspect が前提）。データの件数が減る
#  to_int   ： 値を整数に変換する（inspect が前提。不正値があると失敗する）
#  aggregate： 合計を計算する（to_int が前提）
#  save     ： 合計と行数をJSONファイルに保存する（aggregate が前提）
def is_number(value) -> bool:
    return str(value).lstrip("-").isdigit()


def run_operation(operation: str, state: dict) -> None:
    if operation == "inspect":
        state["inspected"] = True
        state["invalid_count"] = sum(not is_number(value) for value in state["amounts"])
    elif operation == "fill_zero":
        if not state.get("inspected"):
            raise ValueError("fill_zero の前に inspect が必要です。")
        state["amounts"] = [value if is_number(value) else 0 for value in state["amounts"]]
        state["invalid_count"] = 0
    elif operation == "drop_invalid":
        if not state.get("inspected"):
            raise ValueError("drop_invalid の前に inspect が必要です。")
        state["amounts"] = [value for value in state["amounts"] if is_number(value)]
        state["invalid_count"] = 0
    elif operation == "to_int":
        if not state.get("inspected"):
            raise ValueError("to_int の前に inspect が必要です。")
        if any(not is_number(value) for value in state["amounts"]):
            raise ValueError("整数に変換できない値があります。")
        state["amounts"] = [int(value) for value in state["amounts"]]
    elif operation == "aggregate":
        if any(type(value) is not int for value in state["amounts"]):
            raise ValueError("aggregate の前に to_int が必要です。")
        state["total"] = sum(state["amounts"])
    elif operation == "save":
        if "total" not in state:
            raise ValueError("save の前に aggregate が必要です。")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump({"total": state["total"], "rows": len(state["amounts"])}, file, ensure_ascii=False)
        state["saved"] = True
    else:
        raise ValueError(f"未知の操作です：{operation}")


# 保存したファイルを読み、達成条件と同じかを確かめる
def check_goal(expected: dict) -> bool:
    if not os.path.exists(OUTPUT_PATH):
        return False
    with open(OUTPUT_PATH, encoding="utf-8") as file:
        return json.load(file) == expected



#####################################
# LLMのレスポンス形式を定義する
#####################################
Operation = Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]


# 1. 分解：大目標を、小目標に分けたもの
class Subgoals(BaseModel):
    subgoals: list[str] = Field(description="実行する順に並べた小目標（2個）")


# 1. 分解：小目標を、具体的な操作にしたもの
class Steps(BaseModel):
    steps: list[Operation]


# 2. 依存関係：各操作と、その前に終わっている必要がある操作
class DependentStep(BaseModel):
    name: Literal["fetch_sales", "fetch_customers", "join", "save"]
    depends_on: list[Literal["fetch_sales", "fetch_customers", "join", "save"]]


class DependencyPlan(BaseModel):
    steps: list[DependentStep]


# 3. 複数案：不正値への対処が異なる、2つの計画
class Candidates(BaseModel):
    plans: list[Steps] = Field(description="不正値への対処が異なる2つの計画")



#####################################
# 計画を作る（3つの例で共通）
#####################################
prompt_plan = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは作業の計画係です。指示に従って計画を作ってください。\n{instruction}"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "{information}"
    )},
]


# schema の形で計画を受け取る（LangChain の with_structured_output に当たる）
def ask_plan(schema, instruction: str, information: dict):
    values = {"instruction": instruction, "information": json.dumps(information, ensure_ascii=False)}
    message = call_llm(llm, format_prompt(prompt_plan, values), response_format=schema)
    return schema.model_validate_json(message.content)


ACTIONS = {
    "inspect"       : "データの内容と不正値の数を確認する。前提なし。",
    "fill_zero"     : "不正値を0に置き換える。データの件数は変わらない。inspect が前提。",
    "drop_invalid"  : "不正値のデータを取り除く。データの件数が減る。inspect が前提。",
    "to_int"        : "値を整数に変換する。inspect が前提で、不正値があると失敗する。",
    "aggregate"     : "合計を計算する。to_int が前提。",
    "save"          : "合計と行数をJSONに保存する。aggregate が前提。",
}


#####################################
# 2. 依存関係の例で使う操作（売上と顧客のデータを結合する）
#####################################
def run_join_operation(name: str, results: dict):
    if name == "fetch_sales":
        return [{"customer_id": "A", "amount": 10}, {"customer_id": "B", "amount": 5}]
    if name == "fetch_customers":
        return {"A": "顧客A", "B": "顧客B"}
    if name == "join":
        return [{"customer_id": row["customer_id"], "amount": row["amount"],
                 "customer_name": results["fetch_customers"][row["customer_id"]]} for row in results["fetch_sales"]]
    if name == "save":
        return f"{len(results['join'])}件を保存しました（教材のため、ファイルには書き込みません）。"
    raise ValueError(f"未知の操作です：{name}")


def main():
    """
    題材：計画の「形」を3つ試す。
         1. 分解    ：大目標を小目標に分け、小目標ごとに具体的な操作を計画する
         2. 依存関係：操作の前後関係を計画し、同時に実行できる操作は並列に実行する
         3. 複数案  ：計画を2案作って両方試し、決めた基準で良い方を選ぶ
    """

    # ##################################################################
    # 1. 分解：大目標 → 小目標 → 具体的な操作
    # ##################################################################
    print("========== 1. 分解 ==========")
    goal = "売上の合計と行数を、JSONファイルに保存する。"
    result_subgoals = ask_plan(Subgoals, "大目標を、実行する順に2つの小目標に分けてください。", {"大目標": goal})
    print(f"小目標：{result_subgoals.subgoals}")

    state = {"amounts": ["10", "20", "5"]}
    for subgoal in result_subgoals.subgoals:
        result_steps = ask_plan(Steps, "今回の小目標だけを、使える操作の並びにしてください。",
                                {"今回の小目標": subgoal, "使える操作": ACTIONS, "現在の状態": state})
        print(f"小目標「{subgoal}」→ 操作：{result_steps.steps}")
        for step in result_steps.steps:
            run_operation(step, state)
    print(f"結果：{'達成' if check_goal({'total': 35, 'rows': 3}) else '未達成'}")

    # ##################################################################
    # 2. 依存関係：前の操作が終わった操作から、同時に実行する
    # ##################################################################
    print("========== 2. 依存関係 ==========")
    result_dependency = ask_plan(
        DependencyPlan,
        "売上に顧客名を付けて保存する計画を、操作と依存関係で作ってください。"
        "fetch_sales と fetch_customers は他に依存せず、join はその2つに、save は join に依存します。",
        {"目標": "売上に顧客名を付けて保存する"},
    )
    for step in result_dependency.steps:
        print(f"  {step.name} ← 依存：{step.depends_on}")

    results = {}
    while len(results) < len(result_dependency.steps):
        # 依存している操作がすべて終わった操作を、今回まとめて実行する
        ready = [step.name for step in result_dependency.steps
                 if step.name not in results and all(name in results for name in step.depends_on)]
        if not ready:
            raise ValueError("実行できる操作がありません。依存関係が循環しています。")
        print(f"  [同時に実行] {ready}")
        with ThreadPoolExecutor() as executor:
            futures = {name: executor.submit(run_join_operation, name, results) for name in ready}
            for name, future in futures.items():
                results[name] = future.result()
    print(f"結果：{results['save']}")

    # ##################################################################
    # 3. 複数案：2つの計画を試し、処理対象のデータが多い方を選ぶ
    #    入力データ ["10", "?", "5"] は3件。不正値 "?" への対処によって、処理対象のデータの件数が変わる
    #      不正値を取り除く（drop_invalid）  → 処理対象のデータは ["10", "5"]    の2件
    #      不正値を0にする（fill_zero）      → 処理対象のデータは ["10", 0, "5"] の3件
    #    合計はどちらも15になるが、処理対象のデータの件数（rows）が違う。ここでは、処理対象のデータが多い方を良い計画とする
    # ##################################################################
    print("========== 3. 複数案 ==========")
    result_candidates = ask_plan(
        Candidates,
        "売上の合計と件数をJSONに保存する計画を2案作ってください。入力には不正値があります。"
        "1案は不正値を取り除く（drop_invalid）、もう1案は不正値を0にする（fill_zero）計画にしてください。",
        {"入力データ": ["10", "?", "5"], "使える操作": ACTIONS},
    )
    scores = []
    for number, candidate in enumerate(result_candidates.plans, start=1):
        state = {"amounts": ["10", "?", "5"]}
        try:
            for step in candidate.steps:
                run_operation(step, state)
        except ValueError as error:
            print(f"  案{number}：{candidate.steps} → 失敗（{error}）")
            continue
        target_count = len(state["amounts"])
        print(f"  案{number}：{candidate.steps} → 合計 {state.get('total')}、処理対象のデータ {target_count}件（入力は3件）")
        scores.append((target_count, number))
    # 選ぶ基準：処理対象のデータが多い方を、より良い計画とする
    if scores:
        best_count, best_number = max(scores)
        print(f"選んだ計画：案{best_number}（処理対象のデータが最も多い：{best_count}件）")
    else:
        print("実行できた計画がありませんでした。")


if __name__ == "__main__":
    main()
