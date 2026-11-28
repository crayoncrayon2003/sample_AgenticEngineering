import json
import os

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


llm = make_llm("minimal")

#####################################
# Stage 1: Extract key points from the review
#####################################
prompt_extract = [
    # user: A message from the application user
    {"role": "user", "content": (
        'Extract the positives and concerns from the following product review as bullet points.\n'
        'Do not add anything that is not in the review.\n\n'
        "{review}"
    )},
]


# Equivalent to LangChain's chain_extract.invoke({...}). Accept prompt values as a dictionary.
def chain_extract(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_extract, values))
    return message.content


#####################################
# Stage 2: Format the key points as JSON
#####################################
prompt_json = [
    # user: A message from the application user
    {"role": "user", "content": (
        'Convert the following key points to JSON with the array keys good and concerns.\n'
        'Output only JSON.\n\n'
        "{points}"
    )},
]


# Equivalent to LangChain's chain_json.invoke({...}). Accept prompt values as a dictionary.
def chain_json(values: dict) -> dict:
    # response_format requests JSON output. Convert the returned string to a dictionary.
    message = call_llm(llm, format_prompt(prompt_json, values), response_format={"type": "json_object"})
    return json.loads(message.content)


#####################################
# A pipeline connecting the two stages
#####################################
# Pass the first stage's output into {points} in the second stage.
def chain_all(values: dict) -> dict:
    return chain_json({"points": chain_extract(values)})


def main():
    """Example: Extract positives and concerns from a product review and format them as JSON."""

    # Product review
    review = (
        'These wireless earbuds have clear sound. For example, surrounding noise does not bother me. '
        'I also like the small charging case, which is easy to carry. '
        'However, the connection sometimes drops when I use just one earbud. The instructions were also hard to understand.'
    )
    print(f'Review: {review}')

    # ##################################################################
    # Method 1: Call each stage in order and inspect intermediate results
    # ##################################################################
    # Step 1: Extract positives and concerns
    result_step1 = chain_extract({"review": review})
    print('--- Stage 1: Extracted key points ---')
    print(result_step1)

    # Step 2: Convert to JSON
    result_step2 = chain_json({"points": result_step1})
    print('--- Stage 2: JSON ---')
    print(result_step2)

    # ##################################################################
    # Method 2: Extract key points and convert to JSON in a combined pipeline
    # ##################################################################
    print('--- Run the combined pipeline ---')
    print(chain_all({"review": review}))


if __name__ == "__main__":
    main()
