import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.graph import END, START, MessagesState, StateGraph

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total_builtin.json")


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
class Todo(BaseModel):
    #  content： 作業の内容
    #  status ： 作業の状態
    content: str = Field(description="作業の内容")
    status: Literal["pending", "in_progress", "completed"] = Field(description="作業の状態")


@tool
def write_todos(todos: list[Todo]) -> str:
    """計画（ToDoリスト）を書く。呼ぶたびに、リスト全体を置き換える。

    Args:
        todos: 作業の一覧
    """
    todo_list[:] = [todo.model_dump() if isinstance(todo, BaseModel) else todo for todo in todos]
    return "ToDoリストを更新しました。"


#####################################
# ツール2：操作を1つ実行する
#####################################
# 04_03_PlanExecute.py では、Pythonが計画を1手ずつ実行した。ここでは、エージェントが自分でこのツールを呼んで実行する
@tool
def run_step(operation: Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]) -> str:
    """売上データに対して、操作を1つ実行し、実行後の状態を返す。

    Args:
        operation: 実行する操作の名前
    """
    try:
        run_operation(operation, work_state)
    except ValueError as error:
        print(f"  [実行] {operation} → 失敗：{error}")
        return f"失敗しました：{error}"
    print(f"  [実行] {operation} → 状態：{work_state}")
    return f"成功しました。現在の状態：{json.dumps(work_state, ensure_ascii=False)}"


#####################################
# LLMにツールを渡す
#####################################
tools_by_name = {"write_todos": write_todos, "run_step": run_step}
# bind_tools：LLMに「使えるツールの一覧」を渡す。LLMは必要に応じてツール呼び出しを返す
llm_with_tools = llm.bind_tools([write_todos, run_step])

# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは作業の計画と実行の係です。\n"
    "1. 最初に write_todos で、目標を達成するための操作を、前提条件を満たす順に並べた計画を日本語で書く。\n"
    "2. 計画の順に run_step で操作を1つずつ実行し、終わった項目は write_todos で completed にする。\n"
    f"3. run_step は1回ごとにコストが1かかり、予算は{BUDGET}回まで。予算内で終わる計画にする。\n"
    "4. すべて終わったら、行った操作と結果を日本語で簡潔に報告する。"
)


#####################################
# グラフの状態を定義する
#####################################
class AgentState(MessagesState):
    #  messages   ： 会話の履歴（MessagesState が持つ）
    #  step_count ： run_step を実行した回数
    #  stopped    ： 予算を超えて止めたら True
    step_count: int
    stopped: bool


#####################################
# ノード（グラフの1ステップ。状態を受け取り、更新する項目を返す）
#####################################
# LLMを呼ぶ。返事は「回答」か「ツール呼び出しの依頼」のどちらか
def node_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# LLMに依頼されたツールを、1つずつ順番に実行する（操作は順番に意味があるため）
# run_step の実行回数を数え、予算を超えて呼ぼうとしたら止める（LangChain の ToolCallLimitMiddleware に当たる）
def node_tools(state: AgentState) -> dict:
    step_count = state.get("step_count", 0)
    tool_messages = []
    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] == "run_step":
            if step_count >= BUDGET:
                print(f"  [予算超過] run_step は{BUDGET}回までのため、エージェントを止めます。")
                stop = AIMessage(content=f"予算（{BUDGET}回）に達したため、途中で終了しました。")
                return {"messages": tool_messages + [stop], "step_count": step_count, "stopped": True}
            step_count += 1
        tool_messages.append(tools_by_name[tool_call["name"]].invoke(tool_call))
    return {"messages": tool_messages, "step_count": step_count}


# 条件分岐1：ツール呼び出しがあればツールへ。無ければ、それが最終回答なので終了
def select_after_model(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


# 条件分岐2：予算を超えて止めたら終了。そうでなければ、LLMに戻る
def select_after_tools(state: AgentState) -> str:
    if state.get("stopped"):
        return END
    return "model"


#####################################
# エージェント（LLMを呼ぶノードと、ツールを実行するノードを、ループでつないで自分で組む）
#####################################
#  START → model ─┬→ END（ツール呼び出しが無い）
#            ↑    │
#            └ tools ←┘（ツール呼び出しがある）
#               └→ END（予算を超えた）
graph = StateGraph(AgentState)
graph.add_node("model", node_model)
graph.add_node("tools", node_tools)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", select_after_model, ["tools", END])
graph.add_conditional_edges("tools", select_after_tools, ["model", END])
app = graph.compile()


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
    report = app.invoke({"messages": [{"role": "user", "content": request}]}, {"recursion_limit": 50})["messages"][-1].text

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
