import json
import os
import subprocess
import sys
from datetime import datetime

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
GOAL_DIR = os.path.join(ROOT, "outputs", "goal_loop")


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


MAX_ROUNDS = 5

llm = make_llm("low")


#####################################
# Define the LLM response schema (final goal evaluation)
#####################################
class GoalReview(BaseModel):
    # feedback: Evaluation for each goal
    # goals_met: True if all goals are satisfied
    # Put evaluation before the verdict so the model explains before deciding.
    feedback: str = Field(description='Evaluation for each goal in English, including reasons for any unmet goals')
    goals_met: bool = Field(description='true only if the last executed code satisfies all goals')


#####################################
# Monitor: Save generated code and actually execute it
#####################################
# Note: Generated code is actually executed on this PC.
# These teaching safeguards are basic: run in a separate process, omit environment variables such as API keys, and stop after 10 seconds.
def save_and_run(code: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as file:
        file.write(code + "\n")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", path],
            env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return 'Execution failed: Did not finish within 10 seconds.'
    if completed.returncode != 0:
        return f'Execution failed (exit code {completed.returncode}):\n{completed.stderr.strip()[-800:]}'
    return f'Execution succeeded:\n{completed.stdout.strip()[-800:]}'


# Execution records (initialized in main)
run_log = {"dir": "", "round": 0, "last_path": "", "last_result": ""}


#####################################
# Tool: Save and execute code and return the result
#####################################
# In 04_06_GoalLoop.py, a Python for loop repeats generation, execution, and evaluation.
# Here the agent calls this tool to check execution results, then revises and calls it again if needed.
def run_code(code: str) -> str:
    """Accept a complete Python script, save it, execute it, and return the result.

    Args:
        code: Entire Python script without explanations or code fences.
    """
    run_log["round"] += 1
    path = os.path.join(run_log["dir"], f"round_{run_log['round']}.py")
    result = save_and_run(code, path)
    run_log["last_path"] = path
    run_log["last_result"] = result
    print(f"--- Attempt {run_log['round']} ---")
    print(f'Execution output: {result.splitlines()[0]}')
    return result


# system_prompt: The agent's role (both writer and reviewer)
system_prompt = (
    'You are a Python programmer. Write an entire script that meets the request and goals, and '
    'always run it using run_code to check the result.\n'
    'If execution fails or goals are unmet, revise the code and run it again with run_code.\n'
    'Use only the standard library.\n'
    'Finally check the last executed code and output against each goal and return an evaluation. '
    'If execution or an assert fails, the goals are not met.'
)
# Instructions requesting a final evaluation after tool calls finish
review_request = 'Check and evaluate the last executed code and output against each goal.'

# Tool descriptions sent to the LLM
tools = [
    {"type": "function", "function": {
        "name": "run_code",
        "description": 'Accept a complete Python script, save it, execute it, and return the result.',
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": 'The complete Python script to execute (without explanations or code fences)'}},
            "required": ["code"],
        },
    }},
]


#####################################
# Agent: Limit the number of iterations
#####################################
# MAX_ROUNDS: Maximum run_code calls (equivalent to LangChain's ToolCallLimitMiddleware).
# MAX_MODEL_CALLS: Maximum LLM calls, also stopping endless thinking without tools (equivalent to LangChain's ModelCallLimitMiddleware).
MAX_MODEL_CALLS = MAX_ROUNDS * 2 + 2


# Repeat LLM -> tool execution -> LLM until tool calls stop, then request an evaluation.
# If a limit is reached, return None instead of an evaluation.
def run_agent(request: str) -> GoalReview | None:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]
    model_calls = 1
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            if run_log["round"] >= MAX_ROUNDS:
                print(f'  [Limit] Stopping the agent: run_code is limited to {MAX_ROUNDS} calls.')
                return None
            args = json.loads(tool_call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": run_code(**args)})
        if model_calls >= MAX_MODEL_CALLS:
            print(f'  [Limit] Stopping the agent: LLM calls are limited to {MAX_MODEL_CALLS}.')
            return None
        model_calls += 1
        response = call_llm(llm, messages, tools=tools)

    # When tool calls end, request the final evaluation as GoalReview (equivalent to LangChain's response_format).
    messages.append(response)
    messages.append({"role": "user", "content": review_request})
    message = call_llm(llm, messages, response_format=GoalReview)
    return GoalReview.model_validate_json(message.content)


def main():
    """Example: Generate a Python function using the request and goals of 04_06_GoalLoop.py and the flow of 01_LangChain/04_06_GoalLoop_Builtin.py.
    Difference: 04_06_GoalLoop.py uses a Python loop for writer (LLM) -> execution -> reviewer (LLM), up to five rounds. Here the agent executes via run_code, checks output, and revises. Limits on run_code and LLM calls bound the loop; the agent returns a final GoalReview.
    Flow: 1. Write and run code with run_code. 2. Check results and revise/re-execute if needed (up to five times). 3. Return a final evaluation per goal (GoalReview). 4. Python accepts the result only if the last execution succeeded and all goals were judged met. Save each round's code in outputs/goal_loop/.
    """

    use_case = (
        'Create binary_gap(n), which accepts a positive integer and returns the maximum number of consecutive zeros enclosed by ones in its binary representation. '
        'Return zero if there is no such run of zeros.'
    )
    goals = [
        'Accept a positive integer and return the result according to the definition.',
        'Raise TypeError for non-integers (including bool) and ValueError for integers of zero or less.',
        'Include asserts checking 9->2, 529->4, 20->1, 15->0, 32->0, and 1->0.',
        "Display usage examples and results inside if __name__ == '__main__':.",
    ]
    print('Goals:')
    for goal in goals:
        print(f'  - {goal}')

    # Directory storing each round's code
    run_log.update({
        "dir": os.path.join(GOAL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + "_builtin"),
        "round": 0, "last_path": "", "last_result": "",
    })
    os.makedirs(run_log["dir"], exist_ok=True)

    goals_text = "\n".join(f"- {goal}" for goal in goals)
    request = f'[Request]\n{use_case}\n\n[Goals]\n{goals_text}'
    review = run_agent(request)

    # If a limit stops execution, there is no final evaluation.
    if review is None:
        print(f"Stopped early at the call limit. Records: {run_log['dir']}")
        return
    print(f"Evaluation: {'Goal achieved' if review.goals_met else 'Not achieved'}")
    print(f"  {review.feedback}")

    if review.goals_met and run_log["last_result"].startswith('Execution succeeded'):
        print(f"All goals met. Accepted code: {run_log['last_path']}")
    else:
        print(f"Goal not achieved. Records: {run_log['dir']}")


if __name__ == "__main__":
    main()
