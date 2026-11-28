import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total_builtin.json")


def make_llm(effort: str):
    """Create an OpenAI or Claude chat model based on LLM_PROVIDER. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
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
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


llm = make_llm("low")

BUDGET = 5  # Maximum operation count (each operation costs 1)

#####################################
# Operations (plan components): Raise ValueError when prerequisites are not met.
#####################################
# The same operations as 04_03_PlanExecute.py.
# inspect: Check the data and count values that cannot be converted to integers (invalid values).
# fill_zero: Replace invalid values with zero (requires inspect). Preserve the row count.
# drop_invalid: Remove invalid values (requires inspect). Reduce the row count.
# to_int: Convert values to integers (requires inspect; fails if invalid values remain).
# aggregate: Calculate the sum (requires to_int).
# save: Save the sum and row count to a JSON file (requires aggregate).
def is_number(value) -> bool:
    return str(value).lstrip("-").isdigit()


def run_operation(operation: str, state: dict) -> None:
    if operation == "inspect":
        state["inspected"] = True
        state["invalid_count"] = sum(not is_number(value) for value in state["amounts"])
    elif operation == "fill_zero":
        if not state.get("inspected"):
            raise ValueError('inspect is required before fill_zero.')
        state["amounts"] = [value if is_number(value) else 0 for value in state["amounts"]]
        state["invalid_count"] = 0
    elif operation == "drop_invalid":
        if not state.get("inspected"):
            raise ValueError('inspect is required before drop_invalid.')
        state["amounts"] = [value for value in state["amounts"] if is_number(value)]
        state["invalid_count"] = 0
    elif operation == "to_int":
        if not state.get("inspected"):
            raise ValueError('inspect is required before to_int.')
        if any(not is_number(value) for value in state["amounts"]):
            raise ValueError('Some values cannot be converted to integers.')
        state["amounts"] = [int(value) for value in state["amounts"]]
    elif operation == "aggregate":
        if any(type(value) is not int for value in state["amounts"]):
            raise ValueError('to_int is required before aggregate.')
        state["total"] = sum(state["amounts"])
    elif operation == "save":
        if "total" not in state:
            raise ValueError('aggregate is required before save.')
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump({"total": state["total"], "rows": len(state["amounts"])}, file, ensure_ascii=False)
        state["saved"] = True
    else:
        raise ValueError(f'Unknown operation: {operation}')


# Read the saved file and compare it with the success criteria.
def check_goal(expected: dict) -> bool:
    if not os.path.exists(OUTPUT_PATH):
        return False
    with open(OUTPUT_PATH, encoding="utf-8") as file:
        return json.load(file) == expected


# Working data (initialized in main)
work_state: dict = {}


#####################################
# Tool: Execute one operation
#####################################
# In 04_03_PlanExecute.py, Python executes the plan step by step. Here the agent calls this tool itself.
# Combining operations into one tool lets ToolCallLimitMiddleware count all executions (= cost) together.
@tool
def run_step(operation: Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]) -> str:
    """Execute one operation on sales data and return the resulting state.

    Args:
        operation: Name of the operation to execute.
    """
    try:
        run_operation(operation, work_state)
    except ValueError as error:
        print(f'  [Execute] {operation} -> failed: {error}')
        return f'Failed: {error}'
    print(f'  [Execute] {operation} -> state: {work_state}')
    return f'Succeeded. Current state: {json.dumps(work_state, ensure_ascii=False)}'


#####################################
# Built-in middleware 1: Write a plan (ToDo list) and track progress
#####################################
# Equivalent to Plan (structured output) and chain_plan in 04_03_PlanExecute.py.
# The agent receives a write_todos tool; the plan is saved in state["todos"].
todo_list = TodoListMiddleware()


#####################################
# Built-in middleware 2: Limit operation executions to the budget
#####################################
# In 04_03_PlanExecute.py, Python totals and checks the plan cost before execution.
# Here executions are counted while running, and the agent stops when the budget is exceeded (exit_behavior="end").
step_limit = ToolCallLimitMiddleware(tool_name="run_step", run_limit=BUDGET, exit_behavior="end")


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: Agent tools (TodoListMiddleware adds write_todos)
    tools=[run_step],
    # middleware: Plan management and execution limits
    middleware=[todo_list, step_limit],
    # system_prompt: The agent's role
    system_prompt=(
        f'You plan and execute work.\n1. First use write_todos to write a English plan ordering operations to meet the goal and prerequisites.\n2. Execute each operation with run_step in order, marking finished tasks completed with write_todos.\n3. Each run_step costs 1; the budget is {BUDGET} calls. Plan to finish within budget.\n4. Briefly report operations and results in English when finished.'
    ),
)


def main():
    """Example: An agent plans and executes sales aggregation and file output using built-in middleware, for the same task as 04_03_PlanExecute.py.
    Inputs: Sales amounts as strings, e.g., ["10", "20", "5"]. Goal: Save sum and row count as JSON. Operations: inspect, fill_zero, drop_invalid, to_int, aggregate, save (see run_operation for prerequisites). Each costs 1; budget 5.
    Flow: 1. Write a ToDo plan with write_todos. 2. Execute operations with run_step and update the list; ToolCallLimitMiddleware stops the agent if calls exceed five. 3. Python reads saved output and verifies achievement.
    Difference: Budget checking occurs during execution rather than before it, so over-budget plans can partially run.
    """

    # Goal and success criteria
    goal = 'Save the sum and row count of sales to a JSON file.'
    expected = {"total": 35, "rows": 3}
    work_state.clear()
    work_state["amounts"] = ["10", "20", "5"]
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)  # Remove previous output so it cannot falsely satisfy the goal.

    # Operation descriptions (sent to the agent)
    actions = (
        'inspect: Inspect data and count invalid values. No prerequisites.\n'
        'fill_zero: Replace invalid values with zero. Preserve row count. Requires inspect.\n'
        'drop_invalid: Remove invalid values. Reduce row count. Requires inspect.\n'
        'to_int: Convert values to integers. Requires inspect; fails if invalid values remain.\n'
        'aggregate: Calculate the sum. Requires to_int.\n'
        'save: Save sum and row count as JSON. Requires aggregate.'
    )
    request = (
        f'[Goal]\n{goal}\n\n[Available operations and prerequisites]\n{actions}\n\n[Current state]\n{json.dumps(work_state, ensure_ascii=False)}'
    )

    # ##################################################################
    # Steps 1-2: Planning and execution: The agent writes a plan and executes operations one by one.
    # ##################################################################
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})

    print('Plan (final ToDo list state):')
    for todo in result.get("todos", []):
        print(f"  [{todo['status']}] {todo['content']}")
    print(f"Report: {result['messages'][-1].text}")

    # ##################################################################
    # Step 3: Verify achievement: Completing the plan is different from achieving the goal. Inspect the saved contents too.
    # ##################################################################
    if check_goal(expected):
        print(f'Goal achieved. Saved contents: {expected}')
    else:
        print('The plan finished, but the goal was not achieved.')


if __name__ == "__main__":
    main()
