import os
import time
from concurrent.futures import ThreadPoolExecutor

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
# Stage 1: Tasks
#####################################
# A function that creates one task
def make_chain_task(instruction: str):
    prompt_task = [
        # system: Instructions from the application developer
        {"role": "system", "content": instruction + 'Keep the output concise and in English.'},
        # user: A message from the application user
        {"role": "user", "content": 'Topic: {topic}'},
    ]

    # Equivalent to LangChain's chain_task.invoke({...}). Accept prompt values as a dictionary.
    def chain_task(values: dict) -> str:
        return call_llm(llm, format_prompt(prompt_task, values)).content

    return chain_task

# Create three tasks: summary, common misconceptions, and terms.
chain_summary        = make_chain_task('Summarize the topic in three sentences.')
chain_misconceptions = make_chain_task('List two common misconceptions about the topic.')
chain_terms          = make_chain_task('List five important terms for the topic, each on a separate line with a short explanation.')


# Run three tasks concurrently (equivalent to LangChain's RunnableParallel).
def chain_parallel(values: dict) -> dict:
    # ThreadPoolExecutor runs work concurrently in separate threads (Python standard library).
    with ThreadPoolExecutor() as executor:
        future_summary        = executor.submit(chain_summary, values)
        future_misconceptions = executor.submit(chain_misconceptions, values)
        future_terms          = executor.submit(chain_terms, values)
        return {
            "summary": future_summary.result(),
            "misconceptions": future_misconceptions.result(),
            "terms": future_terms.result(),
        }

#####################################
# Stage 2: Combine the results
#####################################
prompt_merge = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'Use the supplied summary, misconceptions, and terms to write a beginner-friendly explanation in English. '
        'Pair each misconception with the correct explanation. Use about 400 characters.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        'Topic: {topic}\n\n[Summary]\n{summary}\n\n'
        '[Common misconceptions]\n{misconceptions}\n\n[Terms]\n{terms}'
    )},
]


# Equivalent to LangChain's chain_merge.invoke({...}). Accept prompt values as a dictionary.
def chain_merge(values: dict) -> str:
    return call_llm(llm, format_prompt(prompt_merge, values)).content


#####################################
# A pipeline connecting the two stages
#####################################
# Pass the first stage's results directly to the second stage.
def chain_all(values: dict) -> str:
    result = chain_parallel(values)
    return chain_merge({**values, **result})


def main():
    """Example: Generate a summary, common misconceptions, and terms concurrently, then combine them into an explanation."""

    topic = 'AI agents'
    print(f'Topic: {topic}')

    # ##################################################################
    # Method 1: Call each stage in order and inspect intermediate results
    # ##################################################################
    start = time.perf_counter()
    result_step1 = chain_parallel({"topic": topic})
    print(f'--- Stage 1: Parallel execution ({time.perf_counter() - start:.1f} seconds) ---')
    print("[summary]")
    print(result_step1["summary"])
    print("[misconceptions]")
    print(result_step1["misconceptions"])
    print("[terms]")
    print(result_step1["terms"])

    result_step2 = chain_merge({"topic": topic, **result_step1})
    print('--- Stage 2: Combined explanation ---')
    print(result_step2)

    # For comparison, run the three tasks in the first stage sequentially.
    start = time.perf_counter()
    chain_summary({"topic": topic})
    chain_misconceptions({"topic": topic})
    chain_terms({"topic": topic})
    print(f'(Reference) Sequential stage 1: {time.perf_counter() - start:.1f} seconds')

    # ##################################################################
    # Method 2: Run the parallel tasks and merge their results in a combined pipeline
    # ##################################################################
    print('--- Run the combined pipeline ---')
    print(chain_all({"topic": topic}))


if __name__ == "__main__":
    main()
