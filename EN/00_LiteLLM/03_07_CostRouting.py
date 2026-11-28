import os
import time

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


#####################################
# Two models: A lightweight model and a more capable model
#####################################
# Prices and speeds vary by provider and time. Check each provider's pricing page for current information.
if PROVIDER == "openai":
    MODEL_NAMES = {"light": "gpt-5-nano", "advanced": "gpt-5-mini"}
else:
    MODEL_NAMES = {"light": "claude-haiku-5-5", "advanced": "claude-sonnet-5-5"}

# Routing criterion: Question length (a simple teaching heuristic, not difficulty itself).
CHARACTER_THRESHOLD = 100

# Replace only the model name in make_llm settings (make_llm also checks the API key).
LLMS = {}
for route, model_name in MODEL_NAMES.items():
    LLMS[route] = make_llm("minimal")
    LLMS[route]["model"] = f"{PROVIDER}/{model_name}"

# system_prompt: Shared role for both models
system_prompt = (
    'You explain AI agents and data processing. '
    'Answer clearly in English. Do not assert guesses about unknown specifications as facts.'
)


# Select a model based on question length.
def select_route(question: str) -> str:
    if len(question) < CHARACTER_THRESHOLD:
        return "light"
    return "advanced"


def main():
    """Example: Route short and long questions to lightweight and more capable models to reduce cost and time. Two questions demonstrate the difference."""

    print(f'Routing criterion: light below {CHARACTER_THRESHOLD} characters; advanced otherwise')

    # Two questions
    questions = [
        'What is ETL?',
        'I want to load multiple CSV files daily, handle different column names and encodings, missing values and duplicates, and save aggregated results as JSON. Account for partial file-read failures. Explain processing order, validation at each stage, retry and resume methods, and design considerations.',
    ]
    for question in questions:
        print("----------------------------------------")
        print(f'Question ({len(question)} characters): {question}')

        # Stage 1: Select a model
        route = select_route(question)
        print(f'Selected route: {route} ({MODEL_NAMES[route]})')

        # Stage 2: Answer with the selected model's agent and record time and usage
        start = time.perf_counter()
        response = litellm.completion(
            model=LLMS[route]["model"],
            reasoning_effort=LLMS[route]["reasoning_effort"],
            max_tokens=LLMS[route]["max_tokens"],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            timeout=120,
            num_retries=1,
        )
        elapsed = time.perf_counter() - start

        answer = response.choices[0].message
        # Token usage is in the response's usage field.
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        print(f"Time: {elapsed:.1f} seconds; tokens: input {usage.get('input_tokens')}, output {usage.get('output_tokens')}")
        print(f'Answer: {answer.content}')


if __name__ == "__main__":
    main()
