import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
# Stage 1: Extract key points from the review
#####################################
prompt_extract = ChatPromptTemplate.from_template(
    'Extract the positives and concerns from the following product review as bullet points.\n'
    'Do not add anything that is not in the review.\n\n'
    "{review}"
)
chain_extract = prompt_extract | llm | StrOutputParser()

#####################################
# Stage 2: Format the key points as JSON
#####################################
prompt_json = ChatPromptTemplate.from_template(
    'Convert the following key points to JSON with the array keys good and concerns.\n'
    'Output only JSON.\n\n'
    "{points}"
)
chain_json = prompt_json | llm | JsonOutputParser()

#####################################
# One chain connecting both stages ---
#####################################
# Pass the first stage's output into {points} in the second prompt.
chain_all = {"points": chain_extract} | chain_json


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
    result_step1 = chain_extract.invoke({"review": review})
    print()
    print('--- Stage 1: Extracted key points ---')
    print(result_step1)

    # Step 2: Convert to JSON
    result_step2 = chain_json.invoke({"points": result_step1})
    print()
    print('--- Stage 2: JSON ---')
    print(result_step2)

    # ##################################################################
    # Method 2: Extract positives and concerns and convert them to JSON in a single chain
    # ##################################################################
    print()
    print('--- Run as one chain ---')
    print(chain_all.invoke({"review": review}))


if __name__ == "__main__":
    main()
