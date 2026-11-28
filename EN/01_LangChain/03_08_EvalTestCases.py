import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
EVAL_DIR = os.path.join(ROOT, "outputs", "evaluations")


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


llm = make_llm("low")

#####################################
# Fictional sales data for testing
#####################################
SALES_DATA = {
    "sales-A": [10, 20, 5],
    "sales-B": [-5, 10, 0],
    "sales-C": [],
}


###############################################################################
# Tool 1: Fetch sales data
###############################################################################
@tool
def fetch_sales(dataset_id: str) -> list[int]:
    """Fetch the specified sales data.

    Args:
        dataset_id: Sales dataset ID, e.g., sales-A.
    """
    return SALES_DATA[dataset_id]


###############################################################################
# Tool 2: Calculate the sum and row count of sales data
###############################################################################
@tool
def aggregate_sales(amounts: list[int]) -> dict:
    """Calculate the sum and row count of sales data. Both are zero for an empty dataset.

    Args:
        amounts: Sales data retrieved by fetch_sales.
    """
    return {"total": sum(amounts), "rows": len(amounts)}


#####################################
# Define the LLM response schema (final answer)
#####################################
class SalesAnswer(BaseModel):
    # dataset_id: Target sales dataset ID
    # total: Sum
    # rows: Row count
    dataset_id: str
    total: int
    rows: int


#####################################
# Agent under evaluation
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[fetch_sales, aggregate_sales],
    # response_format: Final answer schema (SalesAnswer)
    response_format=SalesAnswer,
    # system_prompt: The agent's role
    system_prompt=(
        'You aggregate sales. Retrieve the requested data using fetch_sales, and '
        'pass the returned array unchanged to aggregate_sales. Use each tool exactly once. '
        'Include the dataset ID and the tool-calculated sum and row count in the final answer.'
    ),
)


#####################################
# Evaluation: Compare expected and actual results
#####################################
def evaluate(dataset_id: str, answer: dict, tool_names: list[str]) -> dict:
    amounts = SALES_DATA[dataset_id]
    expected = {"dataset_id": dataset_id, "total": sum(amounts), "rows": len(amounts)}
    checks = {
        'Answer values and ID are correct': answer == expected,
        'Required tools were used exactly once each in the correct order': tool_names == ["fetch_sales", "aggregate_sales"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    """Example: Evaluate an agent using three test cases (sales-A, sales-B, sales-C). Record answers, tool usage, time, and token counts, then save to a file."""

    records = []
    for dataset_id in SALES_DATA:
        print("----------------------------------------")
        print(f'Test: {dataset_id}')

        # Run the agent and measure elapsed time.
        start = time.perf_counter()
        result = agent.invoke({"messages": [{"role": "user", "content": f'Look up the sales sum and row count for {dataset_id}.'}]})
        elapsed = time.perf_counter() - start

        # Extract the final answer, tools used, and token counts from the result.
        answer = result["structured_response"].model_dump()
        tool_names = [call["name"] for m in result["messages"] if m.type == "ai" for call in m.tool_calls]
        tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in result["messages"] if m.type == "ai")

        # Evaluate
        evaluation = evaluate(dataset_id, answer, tool_names)
        print(f'Answer: {answer}')
        print(f'Tools used: {tool_names}')
        print(f"Evaluation: {'Passed' if evaluation['passed'] else 'Failed'} {evaluation['checks']}")
        print(f'Time: {elapsed:.1f} seconds; tokens: {tokens}')

        records.append({
            "dataset_id": dataset_id,
            "answer": answer,
            "tool_names": tool_names,
            "evaluation": evaluation,
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": tokens,
        })

    # Summarize all results and save them to a file.
    passed = sum(record["evaluation"]["passed"] for record in records)
    print("----------------------------------------")
    print(f'Passed: {passed} / {len(records)} cases')

    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"test_cases_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"passed": passed, "total": len(records), "records": records}, file, ensure_ascii=False, indent=2)
    print(f'Saved to: {path}')


if __name__ == "__main__":
    main()
