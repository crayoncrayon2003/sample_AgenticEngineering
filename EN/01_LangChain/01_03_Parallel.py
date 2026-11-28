import os
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

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
# Stage 1: Tasks
#####################################
# A function that creates one task
def make_chain_task(instruction: str):
    prompt_task = ChatPromptTemplate.from_messages([
        # system: Instructions from the application developer
        ("system", instruction + 'Keep the output concise and in English.'),
        # user: A message from the application user
        ("user", 'Topic: {topic}'),
    ])
    chain_task = prompt_task | llm | StrOutputParser()
    return chain_task

# Create three tasks: summary, common misconceptions, and terms.
chain_parallel = RunnableParallel(
    summary       = make_chain_task('Summarize the topic in three sentences.'),
    misconceptions= make_chain_task('List two common misconceptions about the topic.'),
    terms         = make_chain_task('List five important terms for the topic, each on a separate line with a short explanation.'),
)

#####################################
# Stage 2: Combine the results
#####################################
prompt_merge = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'Use the supplied summary, misconceptions, and terms to write a beginner-friendly explanation in English. '
        'Pair each misconception with the correct explanation. Use about 400 characters.',
    ),
    (
        # user: A message from the application user
        "user",
        'Topic: {topic}\n\n[Summary]\n{summary}\n\n'
        '[Common misconceptions]\n{misconceptions}\n\n[Terms]\n{terms}',
    ),
])
chain_merge = prompt_merge | llm | StrOutputParser()

#####################################
# --- One chain connecting both stages ---
#####################################
# Add the first stage's results to input {topic} and pass everything to the second stage.
chain_all = (
    RunnablePassthrough.assign(results=chain_parallel)
    | (lambda x: {"topic": x["topic"], **x["results"]})
    | chain_merge
)


def main():
    """Example: Generate a summary, common misconceptions, and terms concurrently, then combine them into an explanation."""

    topic = 'AI agents'
    print(f'Topic: {topic}')

    # ##################################################################
    # Method 1: Call each stage in order and inspect intermediate results
    # ##################################################################
    start = time.perf_counter()
    result_step1 = chain_parallel.invoke({"topic": topic})
    print()
    print(f'--- Stage 1: Parallel execution ({time.perf_counter() - start:.1f} seconds) ---')
    for key, value in result_step1.items():
        print()
        print(f"[{key}]")
        print(value)

    result_step2 = chain_merge.invoke({"topic": topic, **result_step1})
    print()
    print('--- Stage 2: Combined explanation ---')
    print(result_step2)

    # For comparison, run the three tasks in the first stage sequentially.
    start = time.perf_counter()
    for chain_task in chain_parallel.steps__.values():
        chain_task.invoke({"topic": topic})
    print()
    print(f'(Reference) Sequential stage 1: {time.perf_counter() - start:.1f} seconds')

    # ##################################################################
    # Method 2: Extract positives and concerns and convert them to JSON in a single chain
    # ##################################################################
    print()
    print('--- Run as one chain ---')
    print(chain_all.invoke({"topic": topic}))


if __name__ == "__main__":
    main()
