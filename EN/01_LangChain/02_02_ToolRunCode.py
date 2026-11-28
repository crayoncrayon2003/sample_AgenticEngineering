import ast
import os
import subprocess
import sys
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
CODE_DIR = os.path.join(ROOT, "outputs", "generated_code")
ALLOWED_IMPORTS = {"math", "datetime", "statistics", "itertools", "fractions", "decimal"}


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
# Inspect generated code
#####################################
def check_code(code: str) -> None:
    """Check syntax and imports. Raise ValueError if a problem is found."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f'Module not allowed: {name}')


#########################################################
# Tool 1: Save Python code to a file
#########################################################
@tool
def save_code(code: str) -> str:
    """Save Python code to a .py file and return its filename.

    Args:
        code: Entire Python code to save, without explanations or code fences.
    """
    try:
        check_code(code)
    except (SyntaxError, ValueError) as error:
        return f'Could not save: {error}'

    os.makedirs(CODE_DIR, exist_ok=True)
    filename = f"calc_{uuid4().hex[:8]}.py"
    with open(os.path.join(CODE_DIR, filename), "w", encoding="utf-8") as file:
        file.write(code + "\n")
    print(f'[Saved] {filename}')
    print(code)
    return filename

#########################################################
# Tool 2: Execute Python code at the specified path
#########################################################
@tool
def run_code(filename: str) -> str:
    """Execute the Python file returned by save_code and return standard output.

    Args:
        filename: Filename returned by save_code.
    """
    path = os.path.realpath(os.path.join(CODE_DIR, filename))
    # Do not execute files outside the output directory.
    if os.path.dirname(path) != os.path.realpath(CODE_DIR) or not os.path.isfile(path):
        return 'Execution failed: Specify the filename returned by save_code.'

    try:
        completed = subprocess.run(
            [sys.executable, "-I", path],  # -I: Isolate execution from environment variables and the working directory.
            cwd=CODE_DIR,
            # Do not pass environment variables such as API keys (SYSTEMROOT is required to start Python on Windows).
            env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return 'Execution failed: The program did not finish within 10 seconds.'

    if completed.returncode != 0:
        return f'Execution error:\n{completed.stderr.strip()}'
    print(f'[Execution output] {completed.stdout.strip()}')
    return completed.stdout.strip()


#####################################
# Agent
#####################################
system_prompt = """You handle calculations. Do not calculate mentally; always follow these steps:
                1. Write Python code that prints the result and save it with save_code.
                2. Pass the returned filename to run_code to execute it.
                3. Answer briefly in English based on the output.
                Rules:
                - Only these imports are allowed: {allowed_imports}.
                - On failure, fix and retry once. If it still fails, explain why.
                """

agent = create_agent(
    model=llm,
    tools=[save_code, run_code],
    system_prompt=system_prompt.format(allowed_imports=", ".join(sorted(ALLOWED_IMPORTS))),
)


def main():
    """Example: Solve two calculation problems by generating and executing dedicated Python code."""

    # List of questions
    questions = [
        'Find the sum of all prime numbers from 1 to 1000.',
        'How many days are there from April 1, 2026 to December 25, 2026?',
    ]
    for question in questions:
        print("----------------------------------------")
        print(f'Question: {question}')
        # Run the agent ("user": A message from the application user).
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"Answer: {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
