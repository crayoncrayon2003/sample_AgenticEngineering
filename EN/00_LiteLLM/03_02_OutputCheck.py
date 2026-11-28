import os
import re

import litellm
from dotenv import load_dotenv

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


MAX_ANSWER_LENGTH = 160

llm = make_llm("minimal")

#####################################
# Stage 1: Generate text
#####################################
prompt_answer = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        "You write text. Follow the user's request and create English text within 160 characters. "
        'Do not add facts absent from the request. Return only the generated text, without explanations or Markdown.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        "{request}"
    )},
]

# Equivalent to LangChain's chain_answer.invoke({...}). Accept prompt values as a dictionary.
def chain_answer(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_answer, values))
    return message.content



#####################################
# Stage 2: Check generated text in Python
#####################################
# Return a list of detected issues, or an empty list if there are none.
def inspect_output(answer: str) -> list[str]:
    issues = []
    if not answer.strip():
        issues.append('The answer is empty.')
    if len(answer) > MAX_ANSWER_LENGTH:
        issues.append(f'The answer exceeds {MAX_ANSWER_LENGTH} characters.')
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", answer):
        issues.append('The answer contains an email address.')
    if re.search(r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)", answer):
        issues.append('The answer contains a phone number.')
    return issues


def main():
    """Example: Inspect AI-generated text in Python before returning it. Withhold text with problems. The second of two requests asks to include contact details (email and phone number)."""

    # Two requests
    requests = [
        "Write an announcement saying 'Today's study session starts at 15:00.'",
        "Write an announcement that today's study session starts at 15:00. Include sample@example.com and 03-1234-5678 exactly as written as fictional contact details.",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')

        # Stage 1: Generate text
        result_step1 = chain_answer({"request": request})
        print(f'Generated text: {result_step1}')

        # Stage 2: Check generated text
        result_step2 = inspect_output(result_step1)
        if result_step2:
            for issue in result_step2:
                print(f'Detected issue: {issue}')
            print('Response to the user: The answer did not meet publication requirements and has been withheld.')
        else:
            print(f'Answer returned to the user: {result_step1}')


if __name__ == "__main__":
    main()
