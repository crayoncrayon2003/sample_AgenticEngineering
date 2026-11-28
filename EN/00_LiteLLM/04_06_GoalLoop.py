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
# Define the LLM response schema (goal evaluation)
#####################################
class GoalReview(BaseModel):
    # feedback: Evaluation for each goal and what to fix if unmet
    # goals_met: True if all goals are satisfied
    # Put evaluation before the verdict so the model explains before deciding.
    feedback: str = Field(description='Evaluation for each goal and concrete changes needed for unmet goals')
    goals_met: bool = Field(description='true only if all goals are satisfied')


#####################################
# Writer: Write code that meets the goals
#####################################
prompt_write = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You are a Python programmer. Write an entire Python script that meets the request and goals. '
        'If previous code, execution output, and evaluation are available, use them to revise the code. '
        'Use only the standard library. Output only Python code, without explanations or code fences.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        '[Request]\n{use_case}\n\n[Goals]\n{goals}\n\n[Previous code]\n{code}\n\n'
        '[Previous execution output]\n{run_result}\n\n[Previous evaluation]\n{feedback}'
    )},
]

# Equivalent to LangChain's chain_write.invoke({...}). Accept prompt values as a dictionary.
def chain_write(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_write, values))
    return message.content


#####################################
# Reviewer: Evaluate code and execution output against the goals
#####################################
prompt_review = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You evaluate code. Check the code and actual execution output for each goal. '
        'If execution or an assert fails, the goals are not met.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        '[Request]\n{use_case}\n\n[Goals]\n{goals}\n\n[Code]\n{code}\n\n[Execution output]\n{run_result}'
    )},
]

# Equivalent to LangChain's chain_review.invoke({...}). Accept prompt values as a dictionary.
def chain_review(values: dict) -> GoalReview:
    message = call_llm(llm, format_prompt(prompt_review, values), response_format=GoalReview)
    return GoalReview.model_validate_json(message.content)


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


def main():
    """Example: Ask an LLM to write a Python function, execute it, and revise it until all predefined goals are met.
    Roles: Writer (LLM) generates code from the request and goals, using previous output and feedback on later attempts. Reviewer (LLM) checks code and output against each goal. Python saves and runs the code and passes output to the reviewer.
    Request: binary_gap(n) returns the longest run of zeros enclosed by ones in the binary representation of a positive integer. For example, 9 is 1001, so it returns 2. Goals: 1. Return correct results for positive integers. 2. Raise TypeError for non-integers including bool and ValueError for nonpositive integers. 3. Include asserts for 9->2, 529->4, 20->1, 15->0, 32->0, 1->0. 4. Display examples and results inside if __name__ == '__main__':.
    Flow (up to five rounds): Write code, save and execute it (an assert failure counts as execution failure), evaluate every goal, and stop only when execution succeeds and all goals pass; otherwise repeat. Save code, output, and reviews in outputs/goal_loop/.
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

    # Directory for each round's records (code, execution output, and evaluation)
    run_dir = os.path.join(GOAL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    code = '(none)'
    run_result = '(none)'
    feedback = '(none)'
    goals_text = "\n".join(f"- {goal}" for goal in goals)
    for round_no in range(1, MAX_ROUNDS + 1):
        print(f'--- Attempt {round_no} ---')

        # Step 1: Write code (revise using execution output and evaluation on subsequent rounds).
        code = chain_write({"use_case": use_case, "goals": goals_text, "code": code,
                                   "run_result": run_result, "feedback": feedback})

        # Step 2: Monitor: Save and execute the code.
        code_path = os.path.join(run_dir, f"round_{round_no}.py")
        run_result = save_and_run(code, code_path)
        print(f'Execution output: {run_result.splitlines()[0]}')

        # Step 3: Evaluate against the goals and save records.
        result_review = chain_review({"use_case": use_case, "goals": goals_text, "code": code, "run_result": run_result})
        feedback = result_review.feedback
        print(f"Evaluation: {'Goal achieved' if result_review.goals_met else 'Not achieved'}")
        print(f"  {feedback}")
        with open(os.path.join(run_dir, f"round_{round_no}_review.json"), "w", encoding="utf-8") as file:
            json.dump({"run_result": run_result, "feedback": feedback, "goals_met": result_review.goals_met}, file, ensure_ascii=False, indent=2)

        if result_review.goals_met and run_result.startswith('Execution succeeded'):
            print(f'All goals met. Accepted code: {code_path}')
            return

    print(f'Goal not achieved in {MAX_ROUNDS} rounds. Records: {run_dir}')


if __name__ == "__main__":
    main()
