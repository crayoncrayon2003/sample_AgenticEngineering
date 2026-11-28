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

#####################################
# Define graph state
#####################################
class PlanState(TypedDict, total=False):
    # goal / expected / budget: Goal, success criteria, and budget
    # data: Sales data and operation results (updated by run_operation)
    # steps: Plan created by the LLM
    # cost: Plan cost
    # message: Final result to display
    goal: str
    expected: dict
    budget: int
    data: dict
    steps: list[str]
    cost: int
    message: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Step 1: Planning: The LLM creates an ordered sequence of operations.
def node_plan(state: PlanState) -> dict:
    result_plan = chain_plan.invoke({
        "goal": state["goal"],
        "actions": json.dumps(ACTIONS, ensure_ascii=False),
        "budget": state["budget"],
        "state": json.dumps(state["data"], ensure_ascii=False),
    })
    print(f'Plan: {result_plan.steps} (reason: {result_plan.reason})')
    return {"steps": result_plan.steps}


# Step 2: Budget check: Verify total cost before execution.
def node_check_budget(state: PlanState) -> dict:
    cost = sum(ACTIONS[step]['Cost'] for step in state["steps"])
    print(f"Cost: {cost} (budget: {state['budget']})")
    if cost > state["budget"]:
        return {"cost": cost, "message": 'The plan exceeds the budget and will not run.'}
    return {"cost": cost}


# Step 3: Execution and observation: Execute step by step and display state changes.
def node_execute(state: PlanState) -> dict:
    data = dict(state["data"])
    for step in state["steps"]:
        run_operation(step, data)
        print(f'  [Execute] {step} -> state: {data}')
    return {"data": data}


# Step 4: Verify achievement: Completing a plan does not imply achieving the goal. Inspect the saved contents too.
def node_check_goal(state: PlanState) -> dict:
    if check_goal(state["expected"]):
        return {"message": f"Goal achieved. Saved contents: {state['expected']}"}
    return {"message": 'The plan finished, but the goal was not achieved.'}


# Conditional routing: Execute within budget; end if over budget.
def select_next(state: PlanState) -> str:
    if state["cost"] > state["budget"]:
        return END
    return "execute"


#####################################
# Build the graph
#####################################
# START -> plan -> check_budget -> execute -> check_goal -> END (within budget)
# check_budget -> END (over budget)
graph = StateGraph(PlanState)
graph.add_node("plan", node_plan)
graph.add_node("check_budget", node_check_budget)
graph.add_node("execute", node_execute)
graph.add_node("check_goal", node_check_goal)
graph.add_edge(START, "plan")
graph.add_edge("plan", "check_budget")
graph.add_conditional_edges("check_budget", select_next, ["execute", END])
graph.add_edge("execute", "check_goal")
graph.add_edge("check_goal", END)
app = graph.compile()


def main():
    """Example: Ask the LLM to plan aggregation of sales data and save the result.
    Inputs: Sales amounts stored as strings, e.g., ["10", "20", "5"]. Goal: Save the sum and row count to JSON. Available operations: inspect, fill_zero, drop_invalid, to_int, aggregate, save. Operations have prerequisites (e.g., inspect before to_int). Each costs 1; the budget is 5.
    Flow: 1. The LLM orders operations within budget. 2. Python checks total cost before execution. 3. Python executes each step and displays state changes. 4. Python reads the saved file to verify goal achievement.
    """

    # Goal and success criteria
    inputs = {
        "goal": 'Save the sum and row count of sales to a JSON file.',
        "expected": {"total": 35, "rows": 3},
        "budget": 5,
        "data": {"amounts": ["10", "20", "5"]},
    }
    # Run planning -> budget check -> execution and observation -> achievement verification together.
    result = app.invoke(inputs)
    print(result["message"])


if __name__ == "__main__":
    main()
