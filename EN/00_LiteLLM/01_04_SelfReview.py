import os

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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
prompt_write = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You are a Python programmer. Write code that meets the task requirements. '
        'If previous code and feedback are available, output the entire revised code addressing that feedback. '
        'Output only Python code, without explanations or Markdown code fences.'
    )},
    # user: A message from the application user
    {"role": "user", "content": '[Task]\n{task}\n\n[Previous code]\n{code}\n\n[Feedback]\n{issues}'},
]


# Equivalent to LangChain's chain_write.invoke({...}). Accept prompt values as a dictionary.
def chain_write(values: dict) -> str:
    return call_llm(llm, format_prompt(prompt_write, values)).content


#####################################
# Reviewer
#####################################
prompt_review = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You review code. Check task requirements, bugs, boundary cases, and docstrings. '
        'You cannot execute the code; judge it by reading it.'
    )},
    # user: A message from the application user
    {"role": "user", "content": '[Task]\n{task}\n\n[Code]\n{code}'},
]


# Equivalent to LangChain's chain_review.invoke({...}). Accept prompt values as a dictionary.
def chain_review(values: dict) -> Review:
    message = call_llm(llm, format_prompt(prompt_review, values), response_format=Review)
    return Review.model_validate_json(message.content)


#####################################
# Repeat generation and review
#####################################
def self_review(task: str, MAX_ROUNDS : int) -> str:
    # code: The latest version, updated after each generation.
    # issues: Reviewer feedback, updated after each review.
    # On the first round, there is no previous code or feedback, so pass "(none)".
    code = '(none)'
    issues = '(none)'

    for round_no in range(1, MAX_ROUNDS + 1):
        # Step 1: Generate code (revise according to feedback on subsequent rounds)
        result_write = chain_write({"task": task, "code": code, "issues": issues})
        code = result_write
        print(f'--- Version {round_no} ---')
        print(code)

        # Step 2: Review the code
        result_review = chain_review({"task": task, "code": code})
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
    """Example: Generate a Python function, review it using another prompt, and revise it based on feedback. Repeat until it passes or reaches MAX_ROUNDS."""

    # Task (requirements for the function to generate)
    task = (
        'Create a function named format_duration.\n'
        '1. Accept an integer seconds and return a string in the form "1 day 2 hours 3 minutes 4 seconds".\n'
        '2. Omit units whose value is zero (e.g., 3600 -> "1 hour", 90061 -> "1 day 1 hour 1 minute 1 second").\n'
        '3. If seconds is zero, return "0 seconds".\n'
        '4. Raise TypeError if seconds is not an int. Treat bool as a non-int as well.\n'
        '5. Raise ValueError if seconds is negative.\n'
        '6. Write the docstring in English and include at least three doctest examples.'
    )
    # Repeat generation and review, then receive the final code.
    final_code = self_review(task, 3)
    print('--- Final code ---')
    print(final_code)


if __name__ == "__main__":
    main()
