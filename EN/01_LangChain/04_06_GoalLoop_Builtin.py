import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
GOAL_DIR = os.path.join(ROOT, "outputs", "goal_loop")


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
@tool
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


#####################################
# Built-in middleware: Limit iterations
#####################################
# Equivalent to MAX_ROUNDS (for-loop count) in 04_06_GoalLoop.py.
# ToolCallLimitMiddleware: Limit run_code calls to MAX_ROUNDS.
# ModelCallLimitMiddleware: Also limit LLM calls to stop endless thinking without tools.
# exit_behavior="end": End the agent when a limit is reached instead of raising an exception.
code_limit = ToolCallLimitMiddleware(tool_name="run_code", run_limit=MAX_ROUNDS, exit_behavior="end")
model_limit = ModelCallLimitMiddleware(run_limit=MAX_ROUNDS * 2 + 2, exit_behavior="end")


#####################################
# Agent: Both writer and reviewer
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[run_code],
    # middleware: Iteration limits
    middleware=[code_limit, model_limit],
    # response_format: Final answer schema (final evaluation against goals)
    response_format=GoalReview,
    # system_prompt: The agent's role
    system_prompt=(
        'You are a Python programmer. Write an entire script that meets the request and goals, and '
        'always run it using run_code to check the result.\n'
        'If execution fails or goals are unmet, revise the code and run it again with run_code.\n'
        'Use only the standard library.\n'
        'Finally check the last executed code and output against each goal and return an evaluation. '
        'If execution or an assert fails, the goals are not met.'
    ),
)


def main():
    """Example: Generate a Python function using the same request and goals as 04_06_GoalLoop.py.
    Difference: That file uses a Python for loop for writer (LLM) -> execution -> reviewer (LLM), up to five rounds. Here the agent executes and checks code using run_code, revises it, and returns GoalReview. Built-in ToolCallLimit and ModelCallLimit middleware bound the loop.
    Flow: Write and execute code, revise and retry if goals are unmet (up to five times), then return evaluation per goal. Python accepts only if the last execution succeeded and all goals were judged met. Save each round's code in outputs/goal_loop/.
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
    result = agent.invoke({"messages": [{"role": "user", "content": f'[Request]\n{use_case}\n\n[Goals]\n{goals_text}'}]})

    # When a limit stops the agent, no final evaluation (structured_response) is available.
    review = result.get("structured_response")
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
