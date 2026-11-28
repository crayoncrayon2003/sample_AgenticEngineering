import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# Disable NeMo Guardrails telemetry before importing NeMo Guardrails.
os.environ.setdefault("NEMO_GUARDRAILS_NO_USAGE_STATS", "1")
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus

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
# Stage 2: Inspect generated text using output rails (NeMo Guardrails)
#####################################
# NeMo Guardrails defines inspection mechanisms (rails) in YAML configuration.
# rails.output.flows: Output rails. regex check output blocks output matching predefined regular expressions.
# regex_detection: Blocking patterns for email addresses, phone numbers, and text exceeding MAX_ANSWER_LENGTH.
# Configuration can also be written in config.yml and loaded with RailsConfig.from_path.
RAILS_CONFIG = r"""
rails:
  config:
    regex_detection:
      output:
        patterns:
          - '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
          - '(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)'
          - '^[\s\S]{MIN_LENGTH,}$'
  output:
    flows:
      - regex check output
""".replace("MIN_LENGTH", str(MAX_ANSWER_LENGTH + 1))
rails = LLMRails(RailsConfig.from_content(yaml_content=RAILS_CONFIG), llm=llm)


# Return a list of detected issues, or an empty list if there are none.
# check runs only output rails and returns PASSED or BLOCKED; it does not identify the matching pattern.
def inspect_output(answer: str) -> list[str]:
    if not answer.strip():
        return ['The answer is empty.']
    result = rails.check([{"role": "user", "content": ""}, {"role": "assistant", "content": answer}])
    if result.status == RailStatus.BLOCKED:
        return [f"Rail '{result.rail}' blocked output (email, phone, or length over {MAX_ANSWER_LENGTH} characters)."]
    return []

def main():
    """Example: Inspect AI text with NeMo Guardrails output rails before returning it, withholding problematic output. RAILS_CONFIG.regex_detection blocks email addresses, phone numbers, and text over MAX_ANSWER_LENGTH (160) characters.
    Two requests: A 15:00 study-session announcement (release) and the same with fictional contact details (withhold). inspect_output uses regex check output without an LLM. rails.check returns passed/blocked; Python separately checks for empty text.
    Flow: Generate text, inspect it, then release or withhold. Only inspect_output differs from 01_LangChain/03_02_OutputCheck.py, which implements regex checks directly in Python. Requests, generation, and flow are unchanged. The rail does not identify which pattern matched (email, phone, or length).
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
