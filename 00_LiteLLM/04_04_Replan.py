import json
import os
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


MAX_ROUNDS = 3

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
# LLMのレスポンス形式を定義する（計画）
#####################################
class Plan(BaseModel):
    #  reason： 計画の理由（前回の失敗があれば、それへの対処）
    #  steps ： 実行する操作を、実行する順に並べた一覧
    reason: str = Field(description="この計画にした理由。前回の失敗があれば、どう対処したか（日本語で1文）")
    steps: list[Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]]


#####################################
# 計画を作る
#####################################
prompt_plan = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは作業の計画係です。目標を達成するために、使える操作を前提条件を満たす順に並べてください。"
        "操作にはコストがかかります。コストの合計が予算を超えないようにしてください。"
        "前回の失敗が渡された場合は、同じ計画を繰り返さず、失敗の原因に対処した計画にしてください。"
        "不正値は0に置き換えて、行は残してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "【目標】\n{goal}\n\n【使える操作と前提条件・コスト】\n{actions}\n\n【予算】\n{budget}\n\n【現在の状態】\n{state}\n\n【前回の失敗】\n{failure}"
    )},
]

# LangChain の chain_plan.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_plan(values: dict) -> Plan:
    message = call_llm(llm, format_prompt(prompt_plan, values), response_format=Plan)
    return Plan.model_validate_json(message.content)

# 使える操作の説明とコスト（LLMに渡す）
ACTIONS = {
    "inspect"       : {"説明": "データの内容と不正値の数を確認する。前提なし。", "コスト": 1},
    "fill_zero"     : {"説明": "不正値を0に置き換える。データの件数は変わらない。inspect が前提。", "コスト": 1},
    "drop_invalid"  : {"説明": "不正値のデータを取り除く。データの件数が減る。inspect が前提。", "コスト": 1},
    "to_int"        : {"説明": "値を整数に変換する。inspect が前提で、不正値があると失敗する。", "コスト": 1},
    "aggregate"     : {"説明": "合計を計算する。to_int が前提。", "コスト": 1},
    "save"          : {"説明": "合計と行数をJSONに保存する。aggregate が前提。", "コスト": 1},
}


def main():
    """
    題材：
        売上データを集計して、結果をファイルに保存する作業を、LLMに計画させる。
        計画を実行して失敗したら、失敗の内容を見て、計画を作り直す。

    前提：
        入力データ：売上の金額の一覧。整数に変換できない値（"?"）が含まれている。例：["10", "?", "5"]
        目標      ：売上の合計と行数を、JSONファイルに保存する。不正値は0として扱い、行は残す。
        使える操作：inspect（中身を確認する）、fill_zero（不正値を0にする）、drop_invalid（不正値を取り除く）、to_int（整数に変換する）、aggregate（合計する）、save（保存する）
                    操作には前提条件がある（例：to_int の前に inspect が必要）。
                    各操作のコストは1、予算は5。

    流れ：
        1回目：プログラムが用意した暫定の計画（不正値への対処が無い）で実行する。to_int で失敗する。
        2回目：LLMが、失敗の内容を見て計画を作り直し、もう一度実行する。
        毎回、実行の前に、Pythonが計画のコストが予算に収まっているかを確かめる。
        目標を達成するか、上限回数（3回）に達したら終了する。
    """

    goal = "売上の合計と行数を、JSONファイルに保存する。"
    expected = {"total": 15, "rows": 3}
    budget = 5
    failure = "（なし）"
    # 1回目に使う暫定の計画（不正値への対処が無いため、to_int で失敗する）
    initial_steps = ["inspect", "to_int", "aggregate", "save"]

    for round_no in range(1, MAX_ROUNDS + 1):
        print(f"--- 第{round_no}回 ---")
        # 毎回、元のデータから実行し直す
        state = {"amounts": ["10", "?", "5"]}

        # Step1 計画：1回目は暫定の計画を使う。2回目以降は、LLMが前回の失敗を踏まえて計画を作り直す
        if round_no == 1:
            steps = initial_steps
            print(f"計画：{steps}（プログラムが用意した暫定の計画）")
        else:
            result_plan = chain_plan({
                "goal": goal,
                "actions": json.dumps(ACTIONS, ensure_ascii=False),
                "budget": budget,
                "state": json.dumps(state, ensure_ascii=False),
                "failure": failure,
            })
            steps = result_plan.steps
            print(f"計画：{steps}（理由：{result_plan.reason}）")

        # Step2 予算の検査：実行する前に、コストの合計を確かめる。超えていたら、計画を作り直す
        cost = sum(ACTIONS[step]["コスト"] for step in steps)
        print(f"コスト：{cost}（予算：{budget}）")
        if cost > budget:
            failure = f"計画 {steps} のコスト {cost} が予算 {budget} を超えました。"
            print("  [予算超過] 実行しません。")
            continue

        # Step3 実行：失敗したら、失敗の内容を記録して計画を作り直す
        try:
            for step in steps:
                run_operation(step, state)
                print(f"  [実行] {step} → 状態：{state}")
        except ValueError as error:
            failure = f"計画 {steps} の実行中に失敗しました：{error}"
            print(f"  [失敗を観測] {error}")
            continue

        # Step4 達成の確認
        if check_goal(expected):
            print(f"目標を達成しました。保存内容：{expected}")
            return
        failure = f"計画 {steps} は最後まで実行できましたが、保存内容が達成条件 {expected} と違います。"
        print(f"  [未達成を観測] {failure}")

    print("上限回数に達したため、終了します。")


if __name__ == "__main__":
    main()
