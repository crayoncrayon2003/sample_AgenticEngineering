import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total.json")


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
    #  reason： 計画の理由
    #  steps ： 実行する操作を、実行する順に並べた一覧
    reason: str = Field(description="この順番にした理由（日本語で1文）")
    steps: list[Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]]


#####################################
# 計画を作るチェーン
#####################################
prompt_plan = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは作業の計画係です。目標を達成するために、使える操作を前提条件を満たす順に並べてください。"
        "操作にはコストがかかります。コストの合計が予算を超えないようにしてください。"
    ),(
        # user   : 計画の材料（目標・使える操作・予算・現在の状態）
        "user",
        "【目標】\n{goal}\n\n【使える操作と前提条件・コスト】\n{actions}\n\n【予算】\n{budget}\n\n【現在の状態】\n{state}"
    ),
])
chain_plan = prompt_plan | llm.with_structured_output(Plan, method="json_schema")

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

    前提：
        入力データ：売上の金額の一覧。数字が文字列で入っている。例：["10", "20", "5"]
        目標      ：売上の合計と行数を、JSONファイルに保存する。
        使える操作：inspect（中身を確認する）、fill_zero（不正値を0にする）、drop_invalid（不正値を取り除く）、to_int（整数に変換する）、aggregate（合計する）、save（保存する）
                    操作には前提条件がある（例：to_int の前に inspect が必要）。
                    各操作のコストは1、予算は5。

    流れ：
        1. LLMが、予算に収まるように、操作を並べた計画を作る。
        2. Pythonが、計画のコストが予算に収まっているかを、実行前に確かめる。
        3. Pythonが、計画を1手ずつ実行し、状態の変化を表示する。
        4. Pythonが、保存したファイルを読み、目標を達成したかを確かめる。
    """

    # 目標と達成条件
    goal = "売上の合計と行数を、JSONファイルに保存する。"
    expected = {"total": 35, "rows": 3}
    budget = 5
    state = {"amounts": ["10", "20", "5"]}

    # ##################################################################
    # Step1 計画：LLMが、操作を並べた計画を作る
    # ##################################################################
    result_plan = chain_plan.invoke({
        "goal": goal,
        "actions": json.dumps(ACTIONS, ensure_ascii=False),
        "budget": budget,
        "state": json.dumps(state, ensure_ascii=False),
    })
    print(f"計画：{result_plan.steps}（理由：{result_plan.reason}）")

    # ##################################################################
    # Step2 予算の検査：実行する前に、コストの合計を確かめる
    # ##################################################################
    cost = sum(ACTIONS[step]["コスト"] for step in result_plan.steps)
    print(f"コスト：{cost}（予算：{budget}）")
    if cost > budget:
        print("予算を超えるため、実行しません。")
        return

    # ##################################################################
    # Step3 実行と観測：1手ずつ実行し、状態の変化を表示する
    # ##################################################################
    for step in result_plan.steps:
        run_operation(step, state)
        print(f"  [実行] {step} → 状態：{state}")

    # ##################################################################
    # Step4 達成の確認：計画が終わったことと、目標を達成したことは別。保存内容まで確かめる
    # ##################################################################
    if check_goal(expected):
        print(f"目標を達成しました。保存内容：{expected}")
    else:
        print("計画は終わりましたが、目標は達成できていません。")


if __name__ == "__main__":
    main()
