import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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


llm = make_llm("minimal")

#####################################
# Define the LLM response schema (review results)
#####################################
class Review(BaseModel):
    # passed: True if all requirements are met and no changes are needed.
    # issues: A list of required changes; empty if the code passes.
    passed: bool      = Field(description='true if all requirements are met and no changes are needed')
    issues: list[str] = Field(description='Required changes; an empty list if the code passes')


#####################################
# Writer (both initial generation and revision based on feedback)
#####################################
prompt_write = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You are a Python programmer. Write code that meets the task requirements. '
        'If previous code and feedback are available, output the entire revised code addressing that feedback. '
        'Output only Python code, without explanations or Markdown code fences.'
    ),(
        # user: A message from the application user
        "user",
        '[Task]\n{task}\n\n[Previous code]\n{code}\n\n[Feedback]\n{issues}'
    ),
])
chain_write = prompt_write | llm | StrOutputParser()

#####################################
# Reviewer
#####################################
prompt_review = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You review code. Check task requirements, bugs, boundary cases, and docstrings. '
        'You cannot execute the code; judge it by reading it.',
    ),
    (
        # user: A message from the application user
        "user",
        '[Task]\n{task}\n\n[Code]\n{code}'),
])
chain_review = prompt_review | llm.with_structured_output(Review, method="json_schema")


#####################################
# Tasks (individual workflow steps)
#####################################
# Writer: Generate code (revise according to feedback on later rounds).
@task
def write(task_text: str, code: str, issues: str) -> str:
    return chain_write.invoke({"task": task_text, "code": code, "issues": issues})


# Reviewer: Review code
@task
def review(task_text: str, code: str) -> Review:
    return chain_review.invoke({"task": task_text, "code": code})


#####################################
# Workflow: Repeat generation and review
#####################################
@entrypoint()
def self_review(inputs: dict) -> str:
    # inputs: Task (task) and iteration limit (max_rounds)
    task_text = inputs["task"]
    max_rounds = inputs["max_rounds"]

    # code: The latest version, updated after each generation.
    # issues: Reviewer feedback, updated after each review.
    # On the first round, there is no previous code or feedback, so pass "(none)".
    code = '(none)'
    issues = '(none)'

    for round_no in range(1, max_rounds + 1):
        # Step 1: Generate code (revise according to feedback on subsequent rounds)
        result_write = write(task_text, code, issues).result()
        code = result_write
        print(f'--- Version {round_no} ---')
        print(code)

        # Step 2: Review the code
        result_review = review(task_text, code).result()
        if result_review.passed:
            print('Review: Passed')
            return code

        print('Review: Changes required')
        for issue in result_review.issues:
            print(f"  - {issue}")
        issues = "\n".join(result_review.issues)

    print('The round limit was reached. Finish with the latest version.')
    return code


def main():
    """Example: Generate a Python function, review it with another prompt, and revise according to feedback until it passes or reaches three rounds."""

    # Task (requirements for the function to generate)
    task_text = (
        'Create a function named format_duration.\n'
        '1. Accept an integer seconds and return a string in the form "1 day 2 hours 3 minutes 4 seconds".\n'
        '2. Omit units whose value is zero (e.g., 3600 -> "1 hour", 90061 -> "1 day 1 hour 1 minute 1 second").\n'
        '3. If seconds is zero, return "0 seconds".\n'
        '4. Raise TypeError if seconds is not an int. Treat bool as a non-int as well.\n'
        '5. Raise ValueError if seconds is negative.\n'
        '6. Write the docstring in English and include at least three doctest examples.'
    )
    # Repeat generation and review, then receive the final code.
    final_code = self_review.invoke({"task": task_text, "max_rounds": 3})
    print('--- Final code ---')
    print(final_code)


if __name__ == "__main__":
    main()
