import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

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
# Planning chain (replan using failure records)
#####################################
prompt_plan = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You plan work. Arrange available operations in an order that satisfies prerequisites and achieves the goal. '
        'Operations have costs. Keep the total cost within the budget. '
        'If a previous failure is supplied, address its cause instead of repeating the same plan. '
        'Replace invalid values with zero and preserve their rows.'
    ),(
        # user: Planning inputs (goal, available operations, budget, current state, and previous failure)
        "user",
        '[Goal]\n{goal}\n\n[Available operations, prerequisites, and costs]\n{actions}\n\n[Budget]\n{budget}\n\n[Current state]\n{state}\n\n[Previous failure]\n{failure}'
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

#####################################
# Define graph state
#####################################
class ReplanState(TypedDict, total=False):
    # goal / expected / budget: Goal, success criteria, and budget
    # round_no: Attempt number
    # data: Sales data and operation results (updated by run_operation)
    # steps: Current plan
    # failure: Previous failure details (sent to the LLM when replanning)
    # ok: True if the current stage succeeded (used to proceed or return to planning)
    # message: Final result to display
    goal: str
    expected: dict
    budget: int
    round_no: int
    data: dict
    steps: list[str]
    failure: str
    ok: bool
    message: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Step 1: Planning: Use the provisional plan first; on later attempts, let the LLM replan based on the previous failure.
def node_plan(state: ReplanState) -> dict:
    round_no = state.get("round_no", 0) + 1
    print(f'--- Attempt {round_no} ---')
    # Restart each attempt from the original data.
    data = {"amounts": ["10", "?", "5"]}
    if round_no == 1:
        # Initial provisional plan (to_int fails because invalid values are not handled)
        steps = ["inspect", "to_int", "aggregate", "save"]
        print(f'Plan: {steps} (provisional plan supplied by the program)')
    else:
        result_plan = chain_plan.invoke({
            "goal": state["goal"],
            "actions": json.dumps(ACTIONS, ensure_ascii=False),
            "budget": state["budget"],
            "state": json.dumps(data, ensure_ascii=False),
            "failure": state.get("failure", '(none)'),
        })
        steps = result_plan.steps
        print(f'Plan: {steps} (reason: {result_plan.reason})')
    return {"round_no": round_no, "data": data, "steps": steps}


# Step 2: Budget check
def node_check_budget(state: ReplanState) -> dict:
    cost = sum(ACTIONS[step]['Cost'] for step in state["steps"])
    print(f"Cost: {cost} (budget: {state['budget']})")
    if cost > state["budget"]:
        print('  [Over budget] Will not execute.')
        return {"ok": False, "failure": f"Plan {state['steps']} costs {cost}, exceeding budget {state['budget']}."}
    return {"ok": True}


# Step 3: Execution: Record failure details if execution fails.
def node_execute(state: ReplanState) -> dict:
    data = dict(state["data"])
    try:
        for step in state["steps"]:
            run_operation(step, data)
            print(f'  [Execute] {step} -> state: {data}')
    except ValueError as error:
        print(f'  [Failure observed] {error}')
        return {"ok": False, "data": data, "failure": f"Plan {state['steps']} failed during execution: {error}"}
    return {"ok": True, "data": data}


# Step 4: Verify achievement
def node_check_goal(state: ReplanState) -> dict:
    if check_goal(state["expected"]):
        return {"ok": True, "message": f"Goal achieved. Saved contents: {state['expected']}"}
    failure = f"Plan {state['steps']} completed, but saved contents differ from success criteria {state['expected']}."
    print(f'  [Unmet goal observed] {failure}')
    return {"ok": False, "failure": failure}


# When the attempt limit is reached
def node_give_up(state: ReplanState) -> dict:
    return {"message": 'The attempt limit was reached. Exiting.'}


# Conditional routing: Advance on success; on failure, return to planning until the attempt limit.
def make_select_next(next_node: str):
    def select_next(state: ReplanState) -> str:
        if state["ok"]:
            return next_node
        if state["round_no"] >= MAX_ROUNDS:
            return "give_up"
        return "plan"
    return select_next


#####################################
# Build the graph
#####################################
# START -> plan -> check_budget -> execute -> check_goal -> END (goal achieved)
#            ↑         │             │           │
# On failure, return to planning; at the attempt limit, go to give_up -> END.
graph = StateGraph(ReplanState)
graph.add_node("plan", node_plan)
graph.add_node("check_budget", node_check_budget)
graph.add_node("execute", node_execute)
graph.add_node("check_goal", node_check_goal)
graph.add_node("give_up", node_give_up)
graph.add_edge(START, "plan")
graph.add_edge("plan", "check_budget")
graph.add_conditional_edges("check_budget", make_select_next("execute"), ["execute", "plan", "give_up"])
graph.add_conditional_edges("execute", make_select_next("check_goal"), ["check_goal", "plan", "give_up"])
graph.add_conditional_edges("check_goal", make_select_next(END), [END, "plan", "give_up"])
graph.add_edge("give_up", END)
app = graph.compile()


def main():
    """Example: Ask the LLM to plan sales aggregation and saving, then replan after failure.
    Inputs: Sales amounts including a non-integer value, e.g., ["10", "?", "5"]. Goal: Save sum and row count as JSON, treating invalid values as zero without dropping rows. Operations: inspect, fill_zero, drop_invalid, to_int, aggregate, save. Prerequisites apply (e.g., inspect before to_int). Each costs 1; budget 5.
    First attempt: Run a provisional plan without invalid-value handling; to_int fails. Second attempt: The LLM replans based on the failure and retries. Python checks budget before every attempt. Stop when the goal is achieved or after three attempts.
    """

    inputs = {
        "goal": 'Save the sum and row count of sales to a JSON file.',
        "expected": {"total": 15, "rows": 3},
        "budget": 5,
    }
    # Repeat planning -> budget check -> execution -> achievement verification until success or the attempt limit.
    result = app.invoke(inputs)
    print(result["message"])


if __name__ == "__main__":
    main()
