import json
import os
from typing import Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


def make_llm(effort: str) -> dict:
    """Create LiteLLM model settings based on LLM_PROVIDER. LiteLLM selects the LLM using a 'provider/model' string. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """Call the LLM through LiteLLM and return its message (equivalent to LangChain's llm.invoke)."""
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
    """Fill named placeholders in the prompt with values (equivalent to LangChain's ChatPromptTemplate)."""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# Define the LLM response schema (priority decisions)
#####################################
class TaskPriority(BaseModel):
    # task_id: Task ID
    # reason: Reason for the decision
    # priority: P0 (immediate), P1 (soon), P2 (later)
    task_id: str
    reason: str = Field(description='Reason for the decision (one sentence in English)')
    priority: Literal["P0", "P1", "P2"]


class PriorityPlan(BaseModel):
    # assignments: Priorities for all unstarted tasks
    assignments: list[TaskPriority]


#####################################
# Determine priorities
#####################################
prompt_priority = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You prioritize work. Compare urgency, importance, and deadlines for the current situation and unstarted tasks. '
        'P0 requires immediate action, P1 should be handled soon, and P2 can wait. '
        'If conditions change, such as during an outage, reconsider priorities without being bound by previous ones. '
        'Return one ID, reason, and priority for every unstarted task.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        '[Current situation]\n{situation}\n\n[Unstarted tasks]\n{tasks}'
    )},
]

# Equivalent to LangChain's chain_priority.invoke({...}). Accept prompt values as a dictionary.
def chain_priority(values: dict) -> PriorityPlan:
    message = call_llm(llm, format_prompt(prompt_priority, values), response_format=PriorityPlan)
    return PriorityPlan.model_validate_json(message.content)

# Convert priorities to numbers for sorting.
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}


#####################################
# 1. The LLM assigns priorities to unstarted tasks.
#####################################
def prioritize(tasks: list[dict], situation: str) -> None:
    pending = [task for task in tasks if task["status"] == "pending"]
    result_priority = chain_priority({
        "situation": situation,
        "tasks": json.dumps([{"id": t["id"], "description": t["description"]} for t in pending], ensure_ascii=False),
    })

    # Check in Python whether the LLM's response has missing or extra IDs.
    returned_ids = {assignment.task_id for assignment in result_priority.assignments}
    if returned_ids != {task["id"] for task in pending}:
        raise ValueError(f'Task IDs in the LLM response do not match: {returned_ids}')

    for assignment in result_priority.assignments:
        task = next(task for task in pending if task["id"] == assignment.task_id)
        task["priority"] = assignment.priority
        print(f"  [Priority] {task['id']}: {assignment.priority} ({assignment.reason})")


#####################################
# 2. Python selects and executes the highest-priority task.
#####################################
def run_task(operation: str) -> dict:
    if operation == "summarize_sales":
        amounts = [10, 20, 5]
        return {"total": sum(amounts), "rows": len(amounts)}
    if operation == "check_inventory":
        stocks = {'Product A': 10, 'Product B': 0, 'Product C': 3}
        return {'Out of stock': [name for name, count in stocks.items() if count == 0]}
    if operation == "analyze_incident":
        statuses = [500, 500, 200, 503, 500]
        return {'Request count': len(statuses), 'Server error count': sum(status >= 500 for status in statuses)}
    raise ValueError(f'Unknown processing operation: {operation}')


def run_next_task(tasks: list[dict]) -> None:
    pending = [task for task in tasks if task["status"] == "pending"]
    selected = min(pending, key=lambda task: PRIORITY_ORDER[task["priority"]])
    result = run_task(selected["operation"])
    selected["status"] = "completed"
    print(f"  [Execute] {selected['id']} ({selected['priority']}): {selected['description']} -> {result}")


def main():
    """Example: The LLM prioritizes tasks and Python executes them in priority order. An urgent task arrives while processing two tasks, triggering reprioritization of remaining work."""

    # Initial tasks (two)
    tasks = [
        {
            "id"            : "TASK-001",
            "description"   : 'Calculate the total sales needed by the end of today.',
            "operation"     : "summarize_sales",
            "status"        : "pending",
            "priority"      : None
        },
        {
            "id"            : "TASK-002",
            "description"   : "Find out-of-stock products for next week's regular meeting.",
            "operation"     : "check_inventory",
            "status"        : "pending",
            "priority"      : None
        },
    ]

    # ##################################################################
    # Normal conditions: Prioritize and execute just one task.
    # ##################################################################
    print('--- Normal conditions ---')
    prioritize(tasks, 'Operations are normal. Requests for today and next week have arrived.')
    run_next_task(tasks)

    # ##################################################################
    # An urgent task arrives: Reprioritize remaining tasks and execute all of them.
    # ##################################################################
    print('--- An urgent task arrived ---')
    tasks.append({"id": "TASK-003", "description": 'A service outage has occurred. Aggregate error logs immediately.',
                  "operation": "analyze_incident", "status": "pending", "priority": None})

    prioritize(tasks, 'A service outage has occurred. Reconsider unstarted work.')
    while any(task["status"] == "pending" for task in tasks):
        run_next_task(tasks)


if __name__ == "__main__":
    main()
