import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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

#####################################
# Pass tools to the LLM
#####################################
# bind_tools passes the available tool list to the LLM, which returns tool calls as needed.
llm_with_tools = llm.bind_tools([run_code])
llm_review = llm.with_structured_output(GoalReview, method="json_schema")

# MAX_ROUNDS: Maximum run_code calls (equivalent to LangChain's ToolCallLimitMiddleware).
# MAX_MODEL_CALLS: Maximum LLM calls, also stopping endless thinking without tools (equivalent to LangChain's ModelCallLimitMiddleware).
MAX_MODEL_CALLS = MAX_ROUNDS * 2 + 2


#####################################
# Tasks (individual workflow steps)
#####################################
# Call the LLM. Its response is either an answer or a request to call tools.
@task
def call_model(messages: list) -> AIMessage:
    return llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)


# Execute one tool requested by the LLM.
@task
def call_tool(tool_call: dict) -> ToolMessage:
    return run_code.invoke(tool_call)


# Request the final evaluation as GoalReview (equivalent to LangChain's response_format).
@task
def review(messages: list) -> GoalReview:
    return llm_review.invoke([SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=review_request)])


#####################################
# Agent: Limit iterations. Return None instead of an evaluation when a limit is reached.
#####################################
@entrypoint()
def agent(request: str) -> GoalReview | None:
    messages = [HumanMessage(content=request)]
    model_calls = 1
    response = call_model(messages).result()

    # While the LLM requests tool calls, execute the tools and call the LLM again.
    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if run_log["round"] >= MAX_ROUNDS:
                print(f'  [Limit] Stopping the agent: run_code is limited to {MAX_ROUNDS} calls.')
                return None
            tool_messages.append(call_tool(tool_call).result())
        messages = add_messages(messages, [response] + tool_messages)
        if model_calls >= MAX_MODEL_CALLS:
            print(f'  [Limit] Stopping the agent: LLM calls are limited to {MAX_MODEL_CALLS}.')
            return None
        model_calls += 1
        response = call_model(messages).result()

    # Request a final evaluation when there are no more tool calls.
    return review(add_messages(messages, [response])).result()


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
    review = agent.invoke(request)

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
