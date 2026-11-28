import json
import os
import time
from datetime import datetime

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
EVAL_DIR = os.path.join(ROOT, "outputs", "evaluations")


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


llm = make_llm("low")

#####################################
# Fictional sales data for testing
#####################################
SALES_DATA = {
    "sales-A": [10, 20, 5],
    "sales-B": [-5, 10, 0],
    "sales-C": [],
}


#####################################
# Tool 1: Fetch sales data
#####################################
def fetch_sales(dataset_id: str) -> list[int]:
    """Fetch the specified sales data.

    Args:
        dataset_id: Sales dataset ID, e.g., sales-A.
    """
    return SALES_DATA[dataset_id]


#####################################
# Tool 2: Calculate the sum and row count of sales data
#####################################
def aggregate_sales(amounts: list[int]) -> dict:
    """Calculate the sum and row count of sales data. Both are zero for an empty dataset.

    Args:
        amounts: Sales data retrieved by fetch_sales.
    """
    return {"total": sum(amounts), "rows": len(amounts)}


#####################################
# Tool descriptions sent to the LLM
#####################################
# LangChain's @tool generates this description from the docstring and type hints. Here we write a dictionary manually.
tools = [
    {"type": "function", "function": {
        "name": "fetch_sales",
        "description": 'Fetch the specified sales data.',
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string", "description": 'Sales dataset ID, e.g., sales-A'}},
            "required": ["dataset_id"],
        },
    }},
    {"type": "function", "function": {
        "name": "aggregate_sales",
        "description": 'Calculate the sum and row count of sales data. Both are zero for an empty dataset.',
        "parameters": {
            "type": "object",
            "properties": {"amounts": {"type": "array", "items": {"type": "integer"}, "description": 'Sales data retrieved by fetch_sales'}},
            "required": ["amounts"],
        },
    }},
]
tool_functions = {"fetch_sales": fetch_sales, "aggregate_sales": aggregate_sales}


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
# system_prompt: The agent's role
system_prompt = (
    'You aggregate sales. Retrieve the requested data using fetch_sales, and '
    'pass the returned array unchanged to aggregate_sales. Use each tool exactly once. '
    'Include the dataset ID and the tool-calculated sum and row count in the final answer.'
)


# Run the agent and return the final answer, tools used, and token counts.
def run_agent(message: str) -> tuple[dict, list[str], int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    tool_names = []
    tokens = 0
    while True:
        # Pass both tools and response_format. Once no tools are needed, the answer follows the SalesAnswer schema.
        response = litellm.completion(
            model=llm["model"], reasoning_effort=llm["reasoning_effort"], max_tokens=llm["max_tokens"],
            messages=messages, tools=tools, response_format=SalesAnswer, timeout=60, num_retries=1,
        )
        tokens += response.usage.total_tokens
        message_ai = response.choices[0].message
        if not message_ai.tool_calls:
            return SalesAnswer.model_validate_json(message_ai.content).model_dump(), tool_names, tokens
        messages.append(message_ai)
        for tool_call in message_ai.tool_calls:
            tool_names.append(tool_call.function.name)
            result = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})


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
        answer, tool_names, tokens = run_agent(f'Look up the sales sum and row count for {dataset_id}.')
        elapsed = time.perf_counter() - start

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
