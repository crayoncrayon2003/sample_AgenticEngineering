import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task

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


MAX_ANSWER_LENGTH = 160

llm = make_llm("minimal")

#####################################
# Stage 1: Text generation chain
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        "You write text. Follow the user's request and create English text within 160 characters. "
        'Do not add facts absent from the request. Return only the generated text, without explanations or Markdown.'
    ),(
        # user: A message from the application user
        "user",
        "{request}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


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

#####################################
# Task: Generate text
#####################################
@task
def write_answer(request: str) -> str:
    return chain_answer.invoke({"request": request})


#####################################
# Workflow (describe the control flow in ordinary Python)
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # Stage 1: Generate text
    result_step1 = write_answer(request).result()

    # Stage 2: Inspect generated text and withhold it if problems are found.
    result_step2 = inspect_output(result_step1)
    if result_step2:
        final_answer = 'The answer did not meet publication requirements and has been withheld.'
    else:
        final_answer = result_step1
    return {"draft": result_step1, "issues": result_step2, "final_answer": final_answer}


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

        # Run both stages together: Text generation and inspection.
        result = workflow.invoke(request)
        print(f"Generated text: {result['draft']}")
        for issue in result["issues"]:
            print(f'Detected issue: {issue}')
        print(f"Answer returned to the user: {result['final_answer']}")


if __name__ == "__main__":
    main()
