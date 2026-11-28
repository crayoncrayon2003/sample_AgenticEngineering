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
# LLMのレスポンス形式を定義する（優先度の判断）
#####################################
class TaskPriority(BaseModel):
    #  task_id ： タスクのID
    #  reason  ： 判断の理由
    #  priority： P0（すぐ対応）、P1（早めに対応）、P2（後で対応）
    task_id: str
    reason: str = Field(description="判断の理由（日本語で1文）")
    priority: Literal["P0", "P1", "P2"]


class PriorityPlan(BaseModel):
    #  assignments： 未着手タスクすべての優先度
    assignments: list[TaskPriority]


#####################################
# 優先度を判断する
#####################################
prompt_priority = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは作業の優先度を判断する係です。現在の状況と未着手のタスクについて、緊急性・重要性・期限を比べてください。"
        "P0は直ちに対応、P1は早めに対応、P2は後で対応してよいものです。"
        "障害などで状況が変われば、以前の優先度にとらわれず見直してください。"
        "未着手のタスクすべてについて、ID・理由・優先度を1件ずつ返してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "【現在の状況】\n{situation}\n\n【未着手のタスク】\n{tasks}"
    )},
]

# LangChain の chain_priority.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_priority(values: dict) -> PriorityPlan:
    message = call_llm(llm, format_prompt(prompt_priority, values), response_format=PriorityPlan)
    return PriorityPlan.model_validate_json(message.content)

# 優先度を、並べ替えのための数値にする
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}


#####################################
# 1. 未着手のタスクに、LLMが優先度を付ける
#####################################
def prioritize(tasks: list[dict], situation: str) -> None:
    pending = [task for task in tasks if task["status"] == "pending"]
    result_priority = chain_priority({
        "situation": situation,
        "tasks": json.dumps([{"id": t["id"], "description": t["description"]} for t in pending], ensure_ascii=False),
    })

    # LLMの返答に、IDの不足や余分なIDが無いかを、Pythonで確かめる
    returned_ids = {assignment.task_id for assignment in result_priority.assignments}
    if returned_ids != {task["id"] for task in pending}:
        raise ValueError(f"LLMの返答のタスクIDが合いません：{returned_ids}")

    for assignment in result_priority.assignments:
        task = next(task for task in pending if task["id"] == assignment.task_id)
        task["priority"] = assignment.priority
        print(f"  [優先度] {task['id']}：{assignment.priority}（{assignment.reason}）")


#####################################
# 2. 優先度が最も高いタスクを、Pythonが選んで実行する
#####################################
def run_task(operation: str) -> dict:
    if operation == "summarize_sales":
        amounts = [10, 20, 5]
        return {"total": sum(amounts), "rows": len(amounts)}
    if operation == "check_inventory":
        stocks = {"商品A": 10, "商品B": 0, "商品C": 3}
        return {"在庫切れ": [name for name, count in stocks.items() if count == 0]}
    if operation == "analyze_incident":
        statuses = [500, 500, 200, 503, 500]
        return {"リクエスト数": len(statuses), "サーバーエラー数": sum(status >= 500 for status in statuses)}
    raise ValueError(f"未知の処理です：{operation}")


def run_next_task(tasks: list[dict]) -> None:
    pending = [task for task in tasks if task["status"] == "pending"]
    selected = min(pending, key=lambda task: PRIORITY_ORDER[task["priority"]])
    result = run_task(selected["operation"])
    selected["status"] = "completed"
    print(f"  [実行] {selected['id']}（{selected['priority']}）：{selected['description']} → {result}")


def main():
    """
    題材：LLMがタスクの優先度を判断し、Pythonが優先度の高いものから実行する。
         2件のタスクを処理している途中に、緊急タスクが届き、残りのタスクの優先度を見直す。
    """

    # 最初のタスク（2件）
    tasks = [
        {
            "id"            : "TASK-001",
            "description"   : "本日中に必要な売上合計を集計する。", 
            "operation"     : "summarize_sales",
            "status"        : "pending",
            "priority"      : None
        },
        {
            "id"            : "TASK-002",
            "description"   : "来週の定例会議に向けて、在庫切れの商品を調べる。",
            "operation"     : "check_inventory",
            "status"        : "pending",
            "priority"      : None
        },
    ]

    # ##################################################################
    # 通常時：優先度を付けて、1件だけ実行する
    # ##################################################################
    print("--- 通常時 ---")
    prioritize(tasks, "通常運用中です。今日と来週の依頼が届いています。")
    run_next_task(tasks)

    # ##################################################################
    # 緊急のタスクが届く：残りのタスクの優先度を見直し、すべて実行する
    # ##################################################################
    print("--- 緊急のタスクが届いた ---")
    tasks.append({"id": "TASK-003", "description": "サービス障害が発生したため、直ちにエラーログを集計する。",
                  "operation": "analyze_incident", "status": "pending", "priority": None})

    prioritize(tasks, "サービス障害が発生しました。未着手の仕事を見直してください。")
    while any(task["status"] == "pending" for task in tasks):
        run_next_task(tasks)


if __name__ == "__main__":
    main()
