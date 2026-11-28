import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total.json")


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
# Define the LLM response schema
#####################################
Operation = Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]


# 1. Decomposition: Break the overall goal into subgoals
class Subgoals(BaseModel):
    subgoals: list[str] = Field(description='Two subgoals in execution order')


# 1. Decomposition: Turn each subgoal into concrete operations
class Steps(BaseModel):
    steps: list[Operation]


# 2. Dependencies: Each operation and the operations that must finish before it
class DependentStep(BaseModel):
    name: Literal["fetch_sales", "fetch_customers", "join", "save"]
    depends_on: list[Literal["fetch_sales", "fetch_customers", "join", "save"]]


class DependencyPlan(BaseModel):
    steps: list[DependentStep]


# 3. Alternatives: Two plans with different approaches to invalid values
class Candidates(BaseModel):
    plans: list[Steps] = Field(description='Two plans with different approaches to invalid values')



#####################################
# Create a plan (shared by all three examples)
#####################################
prompt_plan = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You plan work. Follow the instructions to create a plan.\n{instruction}'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        "{information}"
    )},
]


# Receive a plan matching schema (equivalent to LangChain's with_structured_output).
def ask_plan(schema, instruction: str, information: dict):
    values = {"instruction": instruction, "information": json.dumps(information, ensure_ascii=False)}
    message = call_llm(llm, format_prompt(prompt_plan, values), response_format=schema)
    return schema.model_validate_json(message.content)


ACTIONS = {
    "inspect"       : 'Inspect the data and count invalid values. No prerequisites.',
    "fill_zero"     : 'Replace invalid values with zero. Preserve the row count. Requires inspect.',
    "drop_invalid"  : 'Remove invalid values. Reduce the row count. Requires inspect.',
    "to_int"        : 'Convert values to integers. Requires inspect and fails if invalid values remain.',
    "aggregate"     : 'Calculate the sum. Requires to_int.',
    "save"          : 'Save the sum and row count as JSON. Requires aggregate.',
}


#####################################
# 2. Operations for the dependency example (join sales and customer data)
#####################################
def run_join_operation(name: str, results: dict):
    if name == "fetch_sales":
        return [{"customer_id": "A", "amount": 10}, {"customer_id": "B", "amount": 5}]
    if name == "fetch_customers":
        return {"A": 'Customer A', "B": 'Customer B'}
    if name == "join":
        return [{"customer_id": row["customer_id"], "amount": row["amount"],
                 "customer_name": results["fetch_customers"][row["customer_id"]]} for row in results["fetch_sales"]]
    if name == "save":
        return f"Saved {len(results['join'])} records (no actual file writing in this teaching example)."
    raise ValueError(f'Unknown operation: {name}')


def main():
    """Example: Try three plan structures. 1. Decomposition: Split a goal into subgoals and plan concrete steps for each. 2. Dependencies: Plan prerequisites and run independent operations concurrently. 3. Alternatives: Try two plans and choose the better one by a predefined criterion."""

    # ##################################################################
    # 1. Decomposition: Overall goal -> subgoals -> concrete operations
    # ##################################################################
    print('========== 1. Decomposition ==========')
    goal = 'Save the sum and row count of sales to a JSON file.'
    result_subgoals = ask_plan(Subgoals, 'Divide the overall goal into two subgoals in execution order.', {'Overall goal': goal})
    print(f'Subgoals: {result_subgoals.subgoals}')

    state = {"amounts": ["10", "20", "5"]}
    for subgoal in result_subgoals.subgoals:
        result_steps = ask_plan(Steps, 'Arrange available operations for only the current subgoal.',
                                {'Current subgoal': subgoal, 'Available operations': ACTIONS, 'Current state': state})
        print(f"Subgoal '{subgoal}' -> operations: {result_steps.steps}")
        for step in result_steps.steps:
            run_operation(step, state)
    print(f"Result: {'Achieved' if check_goal({'total': 35, 'rows': 3}) else 'Not achieved'}")

    # ##################################################################
    # 2. Dependencies: Concurrently execute operations whose prerequisites are complete
    # ##################################################################
    print('========== 2. Dependencies ==========')
    result_dependency = ask_plan(
        DependencyPlan,
        'Create a plan with operations and dependencies to attach customer names to sales and save them. '
        'fetch_sales and fetch_customers have no dependencies; join depends on both, and save depends on join.',
        {'Goal': 'Attach customer names to sales and save them'},
    )
    for step in result_dependency.steps:
        print(f'  {step.name} <- dependencies: {step.depends_on}')

    results = {}
    while len(results) < len(result_dependency.steps):
        # Execute together all operations whose dependencies have finished.
        ready = [step.name for step in result_dependency.steps
                 if step.name not in results and all(name in results for name in step.depends_on)]
        if not ready:
            raise ValueError('No operation can run. The dependencies contain a cycle.')
        print(f'  [Execute concurrently] {ready}')
        with ThreadPoolExecutor() as executor:
            futures = {name: executor.submit(run_join_operation, name, results) for name in ready}
            for name, future in futures.items():
                results[name] = future.result()
    print(f"Result: {results['save']}")

    # ##################################################################
    # 3. Alternatives: Try two plans and prefer the one retaining more data
    # Input ["10", "?", "5"] has three rows. Handling the invalid "?" changes how many rows are processed.
    # Dropping invalid values (drop_invalid) -> two rows: ["10", "5"]
    # Replacing invalid values with zero (fill_zero) -> three rows: ["10", 0, "5"]
    # Both sums are 15, but row counts differ. Here we prefer the plan that processes more rows.
    # ##################################################################
    print('========== 3. Alternatives ==========')
    result_candidates = ask_plan(
        Candidates,
        'Create two plans to save sales sum and count as JSON. The input contains invalid values. '
        'One plan should drop invalid values (drop_invalid); the other should replace them with zero (fill_zero).',
        {'Input data': ["10", "?", "5"], 'Available operations': ACTIONS},
    )
    scores = []
    for number, candidate in enumerate(result_candidates.plans, start=1):
        state = {"amounts": ["10", "?", "5"]}
        try:
            for step in candidate.steps:
                run_operation(step, state)
        except ValueError as error:
            print(f'  Plan {number}: {candidate.steps} -> failed ({error})')
            continue
        target_count = len(state["amounts"])
        print(f"  Plan {number}: {candidate.steps} -> total {state.get('total')}, {target_count} processed rows (three input rows)")
        scores.append((target_count, number))
    # Selection criterion: Prefer the plan that processes more rows.
    if scores:
        best_count, best_number = max(scores)
        print(f'Selected plan: {best_number} (most processed rows: {best_count})')
    else:
        print('No plan could be executed.')


if __name__ == "__main__":
    main()
