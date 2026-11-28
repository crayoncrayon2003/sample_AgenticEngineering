import os
import time

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
chain_summary        = make_chain_task('Summarize the topic in three sentences.')
chain_misconceptions = make_chain_task('List two common misconceptions about the topic.')
chain_terms          = make_chain_task('List five important terms for the topic, each on a separate line with a short explanation.')

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
# Tasks (individual workflow steps)
#####################################
@task
def summarize(topic: str) -> str:
    return chain_summary.invoke({"topic": topic})


@task
def list_misconceptions(topic: str) -> str:
    return chain_misconceptions.invoke({"topic": topic})


@task
def list_terms(topic: str) -> str:
    return chain_terms.invoke({"topic": topic})


@task
def merge(topic: str, summary: str, misconceptions: str, terms: str) -> str:
    return chain_merge.invoke({"topic": topic, "summary": summary, "misconceptions": misconceptions, "terms": terms})


#####################################
# Workflow (describe the control flow in ordinary Python)
#####################################
@entrypoint()
def workflow(topic: str) -> dict:
    # Stage 1: Start three tasks before waiting with .result(), so they run concurrently.
    start = time.perf_counter()
    future_summary        = summarize(topic)
    future_misconceptions = list_misconceptions(topic)
    future_terms          = list_terms(topic)

    # Collect all three results (wait here for all tasks to finish).
    summary        = future_summary.result()
    misconceptions = future_misconceptions.result()
    terms          = future_terms.result()
    print(f'--- Stage 1: Parallel execution ({time.perf_counter() - start:.1f} seconds) ---')

    # Stage 2: Combine the three results
    answer = merge(topic, summary, misconceptions, terms).result()
    return {"summary": summary, "misconceptions": misconceptions, "terms": terms, "answer": answer}


def main():
    """Example: Generate a summary, common misconceptions, and terms concurrently, then combine them into an explanation."""

    topic = 'AI agents'
    print(f'Topic: {topic}')

    # Run both stages together: Three parallel tasks, then merge.
    start = time.perf_counter()
    result = workflow.invoke(topic)
    print(f'--- Total execution time: {time.perf_counter() - start:.1f} seconds ---')

    print("[summary]")
    print(result["summary"])
    print("[misconceptions]")
    print(result["misconceptions"])
    print("[terms]")
    print(result["terms"])
    print('--- Stage 2: Combined explanation ---')
    print(result["answer"])

    # For comparison, run all work sequentially (three first-stage tasks and the second-stage merge).
    start = time.perf_counter()
    summary = chain_summary.invoke({"topic": topic})
    misconceptions = chain_misconceptions.invoke({"topic": topic})
    terms = chain_terms.invoke({"topic": topic})
    chain_merge.invoke({"topic": topic, "summary": summary, "misconceptions": misconceptions, "terms": terms})
    print(f'(Reference) All work sequentially: {time.perf_counter() - start:.1f} seconds')


if __name__ == "__main__":
    main()
