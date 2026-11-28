import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from guardrails import Guard
from guardrails.settings import settings
from guardrails.validators import FailResult, PassResult, Validator, register_validator

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
# Prepare Guardrails AI
#####################################
# Disable telemetry, which otherwise defaults to enabled when ~/.guardrailsrc is absent.
# Validators use settings from creation time, so disable telemetry before creating them; also disable it through Guard.configure.
settings.rc.enable_metrics = False


#####################################
# Stage 2: Inspect generated text with Guardrails AI's Guard
#####################################
# Validator: One check. _validate returns PassResult on success or FailResult with a reason on failure.
# Register a name with register_validator to add the validator to Guard.
@register_validator(name="sample/not-empty", data_type="string")
class NotEmpty(Validator):
    def _validate(self, value, metadata):
        if not value.strip():
            return FailResult(error_message='The answer is empty.')
        return PassResult()


@register_validator(name="sample/max-length", data_type="string")
class MaxLength(Validator):
    def __init__(self, max_length: int, on_fail=None):
        super().__init__(on_fail=on_fail, max_length=max_length)
        self.max_length = max_length

    def _validate(self, value, metadata):
        if len(value) > self.max_length:
            return FailResult(error_message=f'The answer exceeds {self.max_length} characters.')
        return PassResult()


@register_validator(name="sample/no-email", data_type="string")
class NoEmail(Validator):
    def _validate(self, value, metadata):
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
            return FailResult(error_message='The answer contains an email address.')
        return PassResult()


@register_validator(name="sample/no-phone", data_type="string")
class NoPhone(Validator):
    def _validate(self, value, metadata):
        if re.search(r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)", value):
            return FailResult(error_message='The answer contains a phone number.')
        return PassResult()


# Guard: Run multiple validators together.
# on_fail="noop": Record failures without rewriting text. Other choices include exception and reask (request LLM regeneration).
guard = Guard().use(
    NotEmpty(on_fail="noop"),
    MaxLength(max_length=MAX_ANSWER_LENGTH, on_fail="noop"),
    NoEmail(on_fail="noop"),
    NoPhone(on_fail="noop"),
)
guard.configure(allow_metrics_collection=False)


# Return a list of detected issues, or an empty list if there are none.
def inspect_output(answer: str) -> list[str]:
    outcome = guard.validate(answer)
    return [summary.failure_reason for summary in outcome.validation_summaries or []]

def main():
    """Example: Inspect AI text before returning it, using Guardrails AI Guard and custom validators. Four checks: NotEmpty, MaxLength (160 characters), NoEmail, and NoPhone.
    Requests: A 15:00 study-session announcement (release), and the same with fictional email/phone details (fail and withhold). inspect_output runs all four validators through Guard without an LLM; each returns PassResult or FailResult with a reason. on_fail="noop" leaves text unchanged and collects failure reasons.
    Flow: Generate text, inspect with Guard, then release if no reasons exist; otherwise display reasons and withhold. Only inspect_output differs from 01_LangChain/03_02_OutputCheck.py, which combines checks in one function. Here each check is a validator combined through Guard; requests, generation, flow, and displayed reasons are the same.
    """

    # Two requests
    requests = [
        "Write an announcement saying 'Today's study session starts at 15:00.'",
        "Write an announcement that today's study session starts at 15:00. Include sample@example.com and 03-1234-5678 exactly as written as fictional contact details.",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')

        # Stage 1: Generate text
        result_step1 = chain_answer.invoke({"request": request})
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
