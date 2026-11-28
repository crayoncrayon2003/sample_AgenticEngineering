import json
import os

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
OUTPUT_PATH = os.path.join(ROOT, "outputs", "plans", "sales_total_builtin.json")


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


# Working data and plan (ToDo list), initialized in main.
work_state: dict = {}
todo_list: list = []

OPERATIONS = ["inspect", "fill_zero", "drop_invalid", "to_int", "aggregate", "save"]

#####################################
# Tool 1: Write a plan (ToDo list), equivalent to write_todos added by LangChain's TodoListMiddleware.
#####################################
# Replace the entire list on each call.
def write_todos(todos: list) -> str:
    todo_list[:] = todos
    return 'The ToDo list was updated.'


#####################################
# Tool 2: Execute one operation
#####################################
# In 04_03_PlanExecute.py, Python executes each planned step. Here the agent calls this tool itself.
def run_step(operation: str) -> str:
    try:
        run_operation(operation, work_state)
    except ValueError as error:
        print(f'  [Execute] {operation} -> failed: {error}')
        return f'Failed: {error}'
    print(f'  [Execute] {operation} -> state: {work_state}')
    return f'Succeeded. Current state: {json.dumps(work_state, ensure_ascii=False)}'


# Tool descriptions sent to the LLM
tools = [
    {"type": "function", "function": {
        "name": "write_todos",
        "description": 'Write a plan (ToDo list). Replace the entire list on each call.',
        "parameters": {
            "type": "object",
            "properties": {"todos": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": 'Task description'},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": 'Task status'},
                },
                "required": ["content", "status"],
            }}},
            "required": ["todos"],
        },
    }},
    {"type": "function", "function": {
        "name": "run_step",
        "description": 'Execute one operation on sales data and return the resulting state.',
        "parameters": {
            "type": "object",
            "properties": {"operation": {"type": "string", "enum": OPERATIONS, "description": 'Name of the operation to execute'}},
            "required": ["operation"],
        },
    }},
]
tool_functions = {"write_todos": write_todos, "run_step": run_step}


#####################################
# Agent
#####################################
# system_prompt: The agent's role
system_prompt = (
    f'You plan and execute work.\n1. First use write_todos to write a English plan ordering operations to meet the goal and prerequisites.\n2. Execute each operation with run_step in order, marking finished tasks completed with write_todos.\n3. Each run_step costs 1; the budget is {BUDGET} calls. Plan to finish within budget.\n4. Briefly report operations and results in English when finished.'
)

# Repeat LLM -> tool execution -> LLM until there are no tool calls (equivalent to LangChain's create_agent).
# Count run_step executions and stop calls that would exceed the budget (equivalent to LangChain's ToolCallLimitMiddleware).
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
                    print(f'  [Over budget] Stopping the agent: run_step is limited to {BUDGET} calls.')
                    return f'Stopped early after reaching the budget of {BUDGET} calls.'
                step_count += 1
            result = tool_functions[name](**args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    return response.content


def main():
    """Example: Have an agent plan and execute sales aggregation and file output. Use the same task as 04_03_PlanExecute.py and the flow of 01_LangChain/04_03_PlanExecute_Builtin.py.
    Inputs: Amounts as strings, e.g., ["10", "20", "5"]. Goal: Save sum and row count to JSON. Operations: inspect, fill_zero, drop_invalid, to_int, aggregate, save (see run_operation for prerequisites). Each costs 1; budget 5.
    Flow: 1. The agent writes a ToDo plan with write_todos. 2. It executes operations with run_step and updates the list; attempting more than five run_step calls stops the agent. 3. Python reads the saved file to verify achievement.
    Difference from 04_03_PlanExecute.py: Budget checking occurs during execution, so even an over-budget plan can partially run.
    """

    # Goal and success criteria
    goal = 'Save the sum and row count of sales to a JSON file.'
    expected = {"total": 35, "rows": 3}
    work_state.clear()
    work_state["amounts"] = ["10", "20", "5"]
    todo_list.clear()
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
    report = run_agent(request)

    print('Plan (final ToDo list state):')
    for todo in todo_list:
        print(f"  [{todo['status']}] {todo['content']}")
    print(f'Report: {report}')

    # ##################################################################
    # Step 3: Verify achievement: Completing the plan is different from achieving the goal. Inspect the saved contents too.
    # ##################################################################
    if check_goal(expected):
        print(f'Goal achieved. Saved contents: {expected}')
    else:
        print('The plan finished, but the goal was not achieved.')


if __name__ == "__main__":
    main()
