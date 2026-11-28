import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIDetectionError, PIIMiddleware

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
# Built-in middleware: Stop if the AI's answer contains personal information
#####################################
# 03_02_OutputCheck.py implements its own regular-expression checker (inspect_output).
# PIIMiddleware handles detection and response for each type of personally identifiable information (PII).
# pii_type: Type to detect; built-ins include email, credit_card, ip, mac_address, and url
# detector: Supply a regular expression for types without a built-in detector
# strategy: Response to detection: block (raise an exception), redact, mask (partially hide), or hash
# apply_to_input: Whether to check user input (disabled here because the request contains contact details)
# apply_to_output: Whether to check the AI's answer
pii_email = PIIMiddleware(
    "email",
    strategy="block",
    apply_to_input=False,
    apply_to_output=True,
)
pii_phone = PIIMiddleware(
    "phone",
    detector=r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)",
    strategy="block",
    apply_to_input=False,
    apply_to_output=True,
)

# Convert detected types to English labels for display.
PII_LABELS = {"email": 'Email address', "phone": 'Phone number'}


#####################################
# Agent: Generate text and inspect it with middleware before returning it
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[],
    # middleware: Check personal information in the answer
    middleware=[pii_email, pii_phone],
    # system_prompt: The agent's role
    system_prompt=(
        "You write text. Follow the user's request and create English text within 160 characters. "
        'Do not add facts absent from the request. Return only the generated text, without explanations or Markdown.'
    ),
)


def main():
    """Example: Inspect AI-generated text with built-in middleware before returning it; withhold problematic output. Use the same requests as 03_02_OutputCheck.py; the second asks to include email and phone details. Length checking is not a PII check, so it is handled separately in Python."""

    # Two requests
    requests = [
        "Write an announcement saying 'Today's study session starts at 15:00.'",
        "Write an announcement that today's study session starts at 15:00. Include sample@example.com and 03-1234-5678 exactly as written as fictional contact details.",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')

        try:
            # Run the agent. Personal information in the answer raises PIIDetectionError.
            result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        except PIIDetectionError as error:
            # block stops at the first detected type; even if email and phone are both present, only one type is reported.
            print(f'Detected issue: Contains {PII_LABELS.get(error.pii_type, error.pii_type)} ({len(error.matches)} matches).')
            print('Response to the user: The answer did not meet publication requirements and has been withheld.')
            continue

        answer = result["messages"][-1].text
        print(f'Generated text: {answer}')
        if len(answer) > MAX_ANSWER_LENGTH:
            print(f'Detected issue: The answer exceeds {MAX_ANSWER_LENGTH} characters.')
            print('Response to the user: The answer did not meet publication requirements and has been withheld.')
        else:
            print(f'Answer returned to the user: {answer}')


if __name__ == "__main__":
    main()
