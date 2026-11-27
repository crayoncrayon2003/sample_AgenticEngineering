import json
import os

import litellm
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total_builtin.json")


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

BUDGET = 5  # 操作を実行できる回数（各操作のコストは1）

#####################################
# 操作（計画の部品）：前提条件を満たさないと ValueError を発生させる
#####################################
# 04_03_PlanExecute.py と同じ操作。
#  inspect     ： データの内容と、整数に変換できない値（不正値）の数を確認する
#  fill_zero   ： 不正値を0に置き換える（inspect が前提）。データの件数は変わらない
#  drop_invalid： 不正値のデータを取り除く（inspect が前提）。データの件数が減る
#  to_int      ： 値を整数に変換する（inspect が前提。不正値があると失敗する）
#  aggregate   ： 合計を計算する（to_int が前提）
#  save        ： 合計と行数をJSONファイルに保存する（aggregate が前提）
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


# 作業中のデータと、計画（ToDoリスト）。main で初期化する
work_state: dict = {}
todo_list: list = []

OPERATIONS = ["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]

#####################################
# ツール1：計画（ToDoリスト）を書く（LangChain の TodoListMiddleware が追加する write_todos に当たる）
#####################################
# 呼ぶたびに、リスト全体を置き換える
def write_todos(todos: list) -> str:
    todo_list[:] = todos
    return "ToDoリストを更新しました。"


#####################################
# ツール2：操作を1つ実行する
#####################################
# 04_03_PlanExecute.py では、Pythonが計画を1手ずつ実行した。ここでは、エージェントが自分でこのツールを呼んで実行する
def run_step(operation: str) -> str:
    try:
        run_operation(operation, work_state)
    except ValueError as error:
        print(f"  [実行] {operation} → 失敗：{error}")
        return f"失敗しました：{error}"
    print(f"  [実行] {operation} → 状態：{work_state}")
    return f"成功しました。現在の状態：{json.dumps(work_state, ensure_ascii=False)}"


# LLMに渡すツールの説明
tools = [
    {"type": "function", "function": {
        "name": "write_todos",
        "description": "計画（ToDoリスト）を書く。呼ぶたびに、リスト全体を置き換える。",
        "parameters": {
            "type": "object",
            "properties": {"todos": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "作業の内容"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "作業の状態"},
                },
                "required": ["content", "status"],
            }}},
            "required": ["todos"],
        },
    }},
    {"type": "function", "function": {
        "name": "run_step",
        "description": "売上データに対して、操作を1つ実行し、実行後の状態を返す。",
        "parameters": {
            "type": "object",
            "properties": {"operation": {"type": "string", "enum": OPERATIONS, "description": "実行する操作の名前"}},
            "required": ["operation"],
        },
    }},
]
tool_functions = {"write_todos": write_todos, "run_step": run_step}


#####################################
# エージェント
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは作業の計画と実行の係です。\n"
    "1. 最初に write_todos で、目標を達成するための操作を、前提条件を満たす順に並べた計画を日本語で書く。\n"
    "2. 計画の順に run_step で操作を1つずつ実行し、終わった項目は write_todos で completed にする。\n"
    f"3. run_step は1回ごとにコストが1かかり、予算は{BUDGET}回まで。予算内で終わる計画にする。\n"
    "4. すべて終わったら、行った操作と結果を日本語で簡潔に報告する。"
)

# LLM → ツール実行 → LLM を、ツール呼び出しが無くなるまで繰り返す（LangChain の create_agent に当たる）
# run_step の実行回数を数え、予算を超えて呼ぼうとしたら止める（LangChain の ToolCallLimitMiddleware に当たる）
def run_agent(request: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]
    step_count = 0
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if name == "run_step":
                if step_count >= BUDGET:
                    print(f"  [予算超過] run_step は{BUDGET}回までのため、エージェントを止めます。")
                    return f"予算（{BUDGET}回）に達したため、途中で終了しました。"
                step_count += 1
            result = tool_functions[name](**args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    return response.content


def main():
    """
    題材：
        売上データを集計して、結果をファイルに保存する作業を、エージェントに計画させて実行させる。
        04_03_PlanExecute.py と同じ題材を、01_LangChain/04_03_PlanExecute_Builtin.py と同じ流れで組み立てる。

    前提：
        入力データ：売上の金額の一覧。数字が文字列で入っている。例：["10", "20", "5"]
        目標      ：売上の合計と行数を、JSONファイルに保存する。
        使える操作：inspect・fill_zero・drop_invalid・to_int・aggregate・save（前提条件は run_operation を参照）
                    各操作のコストは1、予算は5。

    流れ：
        1. エージェントが、write_todos で計画（ToDoリスト）を書く。
        2. エージェントが、run_step で操作を1つずつ実行し、ToDoリストを更新する。
           予算（5回）を超えて run_step を呼ぼうとすると、その時点でエージェントを止める。
        3. Pythonが、保存したファイルを読み、目標を達成したかを確かめる。

    04_03_PlanExecute.py との違い：
        予算の検査が「実行する前」ではなく「実行中」になる。予算を超える計画でも途中までは実行される。
    """

    # 目標と達成条件
    goal = "売上の合計と行数を、JSONファイルに保存する。"
    expected = {"total": 35, "rows": 3}
    work_state.clear()
    work_state["amounts"] = ["10", "20", "5"]
    todo_list.clear()
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)  # 前回の結果で、目標を達成したと誤って判定しないように消しておく

    # 操作の説明（エージェントに渡す）
    actions = (
        "inspect：データの内容と不正値の数を確認する。前提なし。\n"
        "fill_zero：不正値を0に置き換える。データの件数は変わらない。inspect が前提。\n"
        "drop_invalid：不正値のデータを取り除く。データの件数が減る。inspect が前提。\n"
        "to_int：値を整数に変換する。inspect が前提で、不正値があると失敗する。\n"
        "aggregate：合計を計算する。to_int が前提。\n"
        "save：合計と行数をJSONに保存する。aggregate が前提。"
    )
    request = (
        f"【目標】\n{goal}\n\n【使える操作と前提条件】\n{actions}\n\n"
        f"【現在の状態】\n{json.dumps(work_state, ensure_ascii=False)}"
    )

    # ##################################################################
    # Step1・2 計画と実行：エージェントが計画を書き、操作を1つずつ実行する
    # ##################################################################
    report = run_agent(request)

    print("計画（ToDoリストの最終状態）：")
    for todo in todo_list:
        print(f"  [{todo['status']}] {todo['content']}")
    print(f"報告：{report}")

    # ##################################################################
    # Step3 達成の確認：計画が終わったことと、目標を達成したことは別。保存内容まで確かめる
    # ##################################################################
    if check_goal(expected):
        print(f"目標を達成しました。保存内容：{expected}")
    else:
        print("計画は終わりましたが、目標は達成できていません。")


if __name__ == "__main__":
    main()
