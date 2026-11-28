import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel
from langfuse import Evaluation, get_client
from langfuse.langchain import CallbackHandler

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


llm = make_llm("low")

#####################################
# Langfuse: Destination for execution records (traces)
#####################################
# Omitting the logging destination sends data to Langfuse Cloud. Send it to local Langfuse instead.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()


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


#####################################
# Langfuse: Register test cases as a dataset and evaluate them as an experiment.
#####################################
DATASET_NAME = "03_08_sales_test_cases"


def register_dataset() -> None:
    langfuse.create_dataset(name=DATASET_NAME)
    for dataset_id in SALES_DATA:
        # Stable IDs overwrite existing test cases instead of duplicating them on repeated runs.
        langfuse.create_dataset_item(dataset_name=DATASET_NAME, id=f"03_08-{dataset_id}", input={"dataset_id": dataset_id})


# Experiment task: Run the agent for one test case
def task(*, item, **kwargs) -> dict:
    dataset_id = item.input["dataset_id"]
    result = agent.invoke({"messages": [{"role": "user", "content": f'Look up the sales sum and row count for {dataset_id}.'}]},
                          config={"callbacks": [CallbackHandler()]})
    answer = result["structured_response"].model_dump()
    tool_names = [call["name"] for m in result["messages"] if m.type == "ai" for call in m.tool_calls]
    return {"answer": answer, "tool_names": tool_names}


# Experiment evaluator: Return evaluate() results as scores (1.0 passed, 0.0 failed).
def score(*, input, output, **kwargs) -> list[Evaluation]:
    evaluation = evaluate(input["dataset_id"], output["answer"], output["tool_names"])
    return [Evaluation(name=name, value=1.0 if ok else 0.0) for name, ok in evaluation["checks"].items()]


def main():
    """Example: Evaluate three sales test cases (sales-A/B/C), recording answers, tool usage, time, and tokens in Langfuse. Register cases as a dataset and run run_experiment; time and tokens are recorded in traces.
    01_LangChain/03_08_EvalTestCases.py stores cases in SALES_DATA, runs and grades them in a for loop, and saves JSON results.
    Here: 1. register_dataset registers only test input data on the Langfuse server, not test code. 2. run_experiment retrieves cases, calls local task (agent execution) and score (grading), and sends traces and scores to the server. 3. Langfuse stores cases and results; it neither runs nor grades tests.
    """

    register_dataset()
    # max_concurrency=1: Run test cases sequentially.
    result = langfuse.get_dataset(DATASET_NAME).run_experiment(
        name='Sales aggregation agent evaluation', task=task, evaluators=[score], max_concurrency=1
    )

    passed = 0
    for item_result in result.item_results:
        print("----------------------------------------")
        print(f"Test: {item_result.item.input['dataset_id']}")
        checks = {e.name: e.value == 1.0 for e in item_result.evaluations}
        print(f"Answer: {item_result.output['answer']}")
        print(f"Tools used: {item_result.output['tool_names']}")
        print(f"Evaluation: {'Passed' if all(checks.values()) else 'Failed'} {checks}")
        passed += all(checks.values())

    # Display the overall results. Inspect them in Langfuse's experiment UI.
    print("----------------------------------------")
    print(f'Passed: {passed} / {len(result.item_results)} cases')
    print(f'Saved to: {result.experiment_url}')

    # Langfuse: Records are batched in the background; flush all remaining records before program exit.
    langfuse.flush()


if __name__ == "__main__":
    main()
