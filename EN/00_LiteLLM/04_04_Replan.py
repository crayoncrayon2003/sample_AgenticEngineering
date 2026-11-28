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


MAX_ROUNDS = 3

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
    # reason: Reason for the plan (including how a previous failure is addressed)
    # steps: Operations in execution order
    reason: str = Field(description='Reason for this plan and how it addresses any previous failure (one sentence in English)')
    steps: list[Literal["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]]


#####################################
# Create a plan
#####################################
prompt_plan = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You plan work. Arrange available operations in an order that satisfies prerequisites and achieves the goal. '
        'Operations have costs. Keep the total cost within the budget. '
        'If a previous failure is supplied, address its cause instead of repeating the same plan. '
        'Replace invalid values with zero and preserve their rows.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        '[Goal]\n{goal}\n\n[Available operations, prerequisites, and costs]\n{actions}\n\n[Budget]\n{budget}\n\n[Current state]\n{state}\n\n[Previous failure]\n{failure}'
    )},
]

# Equivalent to LangChain's chain_plan.invoke({...}). Accept prompt values as a dictionary.
def chain_plan(values: dict) -> Plan:
    message = call_llm(llm, format_prompt(prompt_plan, values), response_format=Plan)
    return Plan.model_validate_json(message.content)

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
    """Example: Ask the LLM to plan sales aggregation and saving, then replan after failure.
    Inputs: Sales amounts including a non-integer value, e.g., ["10", "?", "5"]. Goal: Save sum and row count as JSON, treating invalid values as zero without dropping rows. Operations: inspect, fill_zero, drop_invalid, to_int, aggregate, save. Prerequisites apply (e.g., inspect before to_int). Each costs 1; budget 5.
    First attempt: Run a provisional plan without invalid-value handling; to_int fails. Second attempt: The LLM replans based on the failure and retries. Python checks budget before every attempt. Stop when the goal is achieved or after three attempts.
    """

    goal = 'Save the sum and row count of sales to a JSON file.'
    expected = {"total": 15, "rows": 3}
    budget = 5
    failure = '(none)'
    # Initial provisional plan (to_int fails because invalid values are not handled)
    initial_steps = ["inspect", "to_int", "aggregate", "save"]

    for round_no in range(1, MAX_ROUNDS + 1):
        print(f'--- Attempt {round_no} ---')
        # Restart each attempt from the original data.
        state = {"amounts": ["10", "?", "5"]}

        # Step 1: Planning: Use the provisional plan first; on later attempts, let the LLM replan based on the previous failure.
        if round_no == 1:
            steps = initial_steps
            print(f'Plan: {steps} (provisional plan supplied by the program)')
        else:
            result_plan = chain_plan({
                "goal": goal,
                "actions": json.dumps(ACTIONS, ensure_ascii=False),
                "budget": budget,
                "state": json.dumps(state, ensure_ascii=False),
                "failure": failure,
            })
            steps = result_plan.steps
            print(f'Plan: {steps} (reason: {result_plan.reason})')

        # Step 2: Budget check: Verify total cost before execution; replan if over budget.
        cost = sum(ACTIONS[step]['Cost'] for step in steps)
        print(f'Cost: {cost} (budget: {budget})')
        if cost > budget:
            failure = f'Plan {steps} costs {cost}, exceeding budget {budget}.'
            print('  [Over budget] Will not execute.')
            continue

        # Step 3: Execution: On failure, record it and replan.
        try:
            for step in steps:
                run_operation(step, state)
                print(f'  [Execute] {step} -> state: {state}')
        except ValueError as error:
            failure = f'Plan {steps} failed during execution: {error}'
            print(f'  [Failure observed] {error}')
            continue

        # Step 4: Verify achievement
        if check_goal(expected):
            print(f'Goal achieved. Saved contents: {expected}')
            return
        failure = f'Plan {steps} completed, but saved contents differ from the success criteria {expected}.'
        print(f'  [Unmet goal observed] {failure}')

    print('The attempt limit was reached. Exiting.')


if __name__ == "__main__":
    main()
