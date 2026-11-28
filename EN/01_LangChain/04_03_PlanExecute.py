import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total.json")


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

#####################################
# Operations (plan components): Raise ValueError when prerequisites are not met.
#####################################
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


#####################################
# Define the LLM response schema (plan)
#####################################
class Plan(BaseModel):
    # reason: Reason for the plan
    # steps: Operations in execution order
    reason: str = Field(description='Reason for this order (one sentence in English)')
    steps: list[Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]]


#####################################
# Planning chain
#####################################
prompt_plan = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You plan work. Arrange available operations in an order that satisfies prerequisites and achieves the goal. '
        'Operations have costs. Keep the total cost within the budget.'
    ),(
        # user: Planning inputs (goal, available operations, budget, and current state)
        "user",
        '[Goal]\n{goal}\n\n[Available operations, prerequisites, and costs]\n{actions}\n\n[Budget]\n{budget}\n\n[Current state]\n{state}'
    ),
])
chain_plan = prompt_plan | llm.with_structured_output(Plan, method="json_schema")

# Available operation descriptions and costs (sent to the LLM)
ACTIONS = {
    "inspect"       : {'Description': 'Inspect the data and count invalid values. No prerequisites.', 'Cost': 1},
    "fill_zero"     : {'Description': 'Replace invalid values with zero. Preserve the row count. Requires inspect.', 'Cost': 1},
    "drop_invalid"  : {'Description': 'Remove invalid values. Reduce the row count. Requires inspect.', 'Cost': 1},
    "to_int"        : {'Description': 'Convert values to integers. Requires inspect and fails if invalid values remain.', 'Cost': 1},
    "aggregate"     : {'Description': 'Calculate the sum. Requires to_int.', 'Cost': 1},
    "save"          : {'Description': 'Save the sum and row count as JSON. Requires aggregate.', 'Cost': 1},
}


def main():
    """Example: Ask the LLM to plan aggregation of sales data and save the result.
    Inputs: Sales amounts stored as strings, e.g., ["10", "20", "5"]. Goal: Save the sum and row count to JSON. Available operations: inspect, fill_zero, drop_invalid, to_int, aggregate, save. Operations have prerequisites (e.g., inspect before to_int). Each costs 1; the budget is 5.
    Flow: 1. The LLM orders operations within budget. 2. Python checks total cost before execution. 3. Python executes each step and displays state changes. 4. Python reads the saved file to verify goal achievement.
    """

    # Goal and success criteria
    goal = 'Save the sum and row count of sales to a JSON file.'
    expected = {"total": 35, "rows": 3}
    budget = 5
    state = {"amounts": ["10", "20", "5"]}

    # ##################################################################
    # Step 1: Planning: The LLM creates an ordered sequence of operations.
    # ##################################################################
    result_plan = chain_plan.invoke({
        "goal": goal,
        "actions": json.dumps(ACTIONS, ensure_ascii=False),
        "budget": budget,
        "state": json.dumps(state, ensure_ascii=False),
    })
    print(f'Plan: {result_plan.steps} (reason: {result_plan.reason})')

    # ##################################################################
    # Step 2: Budget check: Verify total cost before execution.
    # ##################################################################
    cost = sum(ACTIONS[step]['Cost'] for step in result_plan.steps)
    print(f'Cost: {cost} (budget: {budget})')
    if cost > budget:
        print('The plan exceeds the budget and will not run.')
        return

    # ##################################################################
    # Step 3: Execution and observation: Execute step by step and display state changes.
    # ##################################################################
    for step in result_plan.steps:
        run_operation(step, state)
        print(f'  [Execute] {step} -> state: {state}')

    # ##################################################################
    # Step 4: Verify achievement: Completing a plan does not imply achieving the goal. Inspect the saved contents too.
    # ##################################################################
    if check_goal(expected):
        print(f'Goal achieved. Saved contents: {expected}')
    else:
        print('The plan finished, but the goal was not achieved.')


if __name__ == "__main__":
    main()
