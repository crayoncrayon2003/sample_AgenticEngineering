import json
import os
import subprocess
import sys

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
SCRUM_DIR = os.path.join(ROOT, "outputs", "scrum")


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
# Product goal and backlog (list of features to build)
#####################################
PRODUCT_GOAL = 'Build a small command-line program that safely aggregates sales data.'
BACKLOG = [
    {"id": "PBI-01", "value": 'Obtain the sum and row count for valid sales data.', "depends_on": []},
    {"id": "PBI-02", "value": 'Treat null and empty strings as zero and preserve the row count.', "depends_on": ["PBI-01"]},  # Prerequisite: PBI-01
    {"id": "PBI-03", "value": 'Return an error when a nonnumeric string is found.', "depends_on": ["PBI-01"]},    # Prerequisite: PBI-01
]
# User feedback arriving after sprint 1
FEEDBACK_AFTER_SPRINT1 = 'Many sales entries are blank, so please prioritize support for null and empty strings next.'

# Rules for the program written by developers
CODE_CONTRACT = """Return an entire executable Python script.
                Read one JSON object {"amounts": [...]} from standard input and output only JSON {"total": integer, "rows": integer}. Handle integers, integer strings, and negative numbers. For an empty array, both sum and row count are zero.
                If PBI-02 is in scope, replace null and empty strings with zero without dropping rows. If PBI-03 is in scope, return JSON {"error": "invalid_amount"} for nonnumeric strings.
                Use only json and sys from the standard library. Do not include explanations or code fences.
                """

# Define the LLM response schema (backlog order decided by the PO)
class BacklogOrder(BaseModel):
    # reason: Reason for this order
    # ordered_ids: IDs of unfinished items, ordered by value
    reason: str = Field(description='Reason for this order (one sentence in English)')
    ordered_ids: list[str]


#####################################
# Roles (one PO, one SM, three developers)
#####################################
# The developers have no hierarchy; all three share responsibility for implementation and quality.
PEOPLE = {
    "PO": 'You are the Product Owner (PO), accountable for value and backlog ordering.',
    "SM": "You are the Scrum Master (SM). Help improve the team's way of working. Do not assign work.",
    'Developer A': 'You are Developer A. Cooperate with the other developers and share responsibility for implementation, reviews, and quality.',
    'Developer B': 'You are Developer B. Cooperate with the other developers and share responsibility for implementation, reviews, and quality.',
    'Developer C': 'You are Developer C. Cooperate with the other developers and share responsibility for implementation, reviews, and quality.',
}

#####################################
# Chain for assigning work to a role (shared by all five participants)
#####################################
prompt_person = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        "{role}\n{instruction}"
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        'Shared information:\n{information}'
    )},
]


# Ask the role named by name. If output is supplied, receive an answer matching that schema.
def ask_person(name: str, instruction: str, information: dict, output=None):
    values = {
        "role": PEOPLE[name],
        "instruction": instruction,
        "information": json.dumps(information, ensure_ascii=False),
    }
    message = call_llm(llm, format_prompt(prompt_person, values), response_format=output)
    if output is None:
        return message.content
    return output.model_validate_json(message.content)


#####################################
# Acceptance tests: Actually run the developer's program to verify it
#####################################
# Note: Generated code is actually executed on this PC.
# These teaching safeguards are basic: run in a separate process, omit environment variables such as API keys, and stop after five seconds.
def test_program(path: str, target_ids: list[str]) -> dict:
    tests = [
        {"name": 'Normal aggregation', "input": ["10", "20", "5"], "expected": {"total": 35, "rows": 3}},
        {"name": 'Empty data', "input": [], "expected": {"total": 0, "rows": 0}},
    ]
    if "PBI-02" in target_ids:
        tests.append({"name": 'Fill blank values', "input": ["10", None, "", "5"], "expected": {"total": 15, "rows": 4}})
    if "PBI-03" in target_ids:
        tests.append({"name": 'Invalid value', "input": ["10", "?"], "expected": {"error": "invalid_amount"}})

    results = []
    for test in tests:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", path],
                input=json.dumps({"amounts": test["input"]}),
                env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
                capture_output=True, text=True, timeout=5,
            )
            actual = json.loads(completed.stdout) if completed.returncode == 0 else completed.stderr.strip()[-200:]
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            actual = str(error)
        results.append({"name": test["name"], "passed": actual == test["expected"], "actual": actual})
    return {"passed": all(result["passed"] for result in results), "tests": results}


# Save generated code (remove code fences if present).
def save_code(code: str, filename: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        code = "\n".join(line for line in lines if not line.startswith("```"))
    os.makedirs(SCRUM_DIR, exist_ok=True)
    path = os.path.join(SCRUM_DIR, filename)
    with open(path, "w", encoding="utf-8") as file:
        file.write(code + "\n")
    return path


def main():
    """Example: Briefly simulate two development sprints with one PO, one SM, and three developers.
    Roles: The PO orders the feature backlog using user feedback and describes delivered value and direction at sprint end. The SM suggests improvements to the team's workflow at sprint end without assigning work. Developers A, B, and C share implementation and quality responsibility, rotating implementation, review, and finishing roles each sprint. The implementer builds the selected item; the reviewer inspects code and test results and suggests fixes; the finisher applies the review.
    Goal: A small CLI for safe sales aggregation. Backlog: PBI-01 valid sales aggregation; PBI-02 treat blanks as zero; PBI-03 reject invalid values. After sprint 1, feedback asks to prioritize blank handling. Python actually runs acceptance tests.
    Sprint flow: 1. PO orders the backlog; select one item whose prerequisites are complete. 2. Implement and test. 3. Review code and tests and suggest fixes. 4. Finish using review and retest; passing completes the item. 5. PO reviews delivered value and direction. 6. SM proposes a workflow improvement for the next sprint.
    """

    done_ids = []
    feedback = 'First, I want to aggregate valid data.'
    accepted_code = '(none)'
    developers = ['Developer A', 'Developer B', 'Developer C']

    for sprint in (1, 2):
        print(f'========== Sprint {sprint} ==========')

        # Rotate the three developers' roles each sprint.
        # Sprint 1: Implementation A, review B, finishing C
        # Sprint 2: Implementation B, review C, finishing A
        implementer = developers[(sprint - 1) % 3]
        reviewer = developers[sprint % 3]
        finisher = developers[(sprint + 1) % 3]
        print(f'[Roles] Implementer: {implementer}; reviewer: {reviewer}; finisher: {finisher}')

        # ##################################################################
        # 1. Planning: The PO orders the backlog; choose one item with completed prerequisites.
        # ##################################################################
        pending = [item for item in BACKLOG if item["id"] not in done_ids]
        result_order = ask_person(
            "PO",
            instruction='Order all unfinished item IDs by value. Give weight to user feedback.',
            information={'Goal': PRODUCT_GOAL, 'Unfinished items': pending, 'User feedback': feedback},
            output=BacklogOrder,
        )
        print(f'[PO] Order: {result_order.ordered_ids} ({result_order.reason})')
        selected = next(item for item_id in result_order.ordered_ids for item in pending
                        if item["id"] == item_id and set(item["depends_on"]) <= set(done_ids))
        target_ids = done_ids + [selected["id"]]
        print(f"[Plan] Selected item: {selected['id']} ({selected['value']})")

        # ##################################################################
        # 2. Implementation: The implementer writes code and Python runs acceptance tests.
        # ##################################################################
        code = ask_person(
            implementer,
            instruction=CODE_CONTRACT,
            information={'Selected item': target_ids, 'Previous code': accepted_code},
        )
        path = save_code(code, f"sprint{sprint}_v1.py")
        result_test = test_program(path, target_ids)
        print(f'[{implementer}] Implementation: {path}')
        print(f"[Test] {'Passed' if result_test['passed'] else 'Failed'}: {[(t['name'], t['passed']) for t in result_test['tests']]}")

        # ##################################################################
        # 3. Review: The reviewer reads code and test results and identifies problems and fixes.
        # ##################################################################
        review = ask_person(
            reviewer,
            instruction='Read the code and test results. Describe problems and proposed fixes in English within 200 characters. If there are no problems, state its strengths.',
            information={'Selected item': target_ids, 'Code': code, 'Test results': result_test},
        )
        print(f'[{reviewer}] Review: {review}')

        # ##################################################################
        # 4. Finishing: The finisher applies the review and Python retests.
        # ##################################################################
        code = ask_person(
            finisher,
            instruction=CODE_CONTRACT + '\nReturn the entire script finalized using the review.',
            information={'Selected item': target_ids, 'Current code': code, 'Test results': result_test, 'Review': review},
        )
        path = save_code(code, f"sprint{sprint}_v2.py")
        result_test = test_program(path, target_ids)
        print(f'[{finisher}] Finishing: {path}')
        print(f"[Test] {'Passed' if result_test['passed'] else 'Failed'}: {[(t['name'], t['passed']) for t in result_test['tests']]}")

        if result_test["passed"]:
            done_ids.append(selected["id"])
            accepted_code = code

        # ##################################################################
        # 5. Sprint review: The PO describes delivered value and next steps.
        # ##################################################################
        feedback = FEEDBACK_AFTER_SPRINT1 if sprint == 1 else 'No additional feedback.'
        po_review = ask_person(
            "PO",
            instruction='Based on test results and user feedback, describe delivered value and next steps in English within 150 characters.',
            information={'Completed items': done_ids, 'Test results': result_test, 'User feedback': feedback},
        )
        print(f'[PO] Sprint review: {po_review}')

        # ##################################################################
        # 6. Retrospective: The SM suggests an improvement to try next sprint.
        # ##################################################################
        retro = ask_person(
            "SM",
            instruction="Based on this sprint's implementation, review, and finishing process, suggest one workflow improvement for the next sprint in English within 100 characters.",
            information={'Review': review, 'Final test results': result_test},
        )
        print(f'[SM] Retrospective: {retro}')

    print(f'Completed items: {done_ids}')


if __name__ == "__main__":
    main()
