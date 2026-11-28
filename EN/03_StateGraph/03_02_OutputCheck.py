import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

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


MAX_ANSWER_LENGTH = 160

llm = make_llm("minimal")

#####################################
# Stage 1: Text generation chain
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        "You write text. Follow the user's request and create English text within 160 characters. "
        'Do not add facts absent from the request. Return only the generated text, without explanations or Markdown.'
    ),(
        # user: A message from the application user
        "user",
        "{request}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# Stage 2: Check generated text in Python
#####################################
# Return a list of detected issues, or an empty list if there are none.
def inspect_output(answer: str) -> list[str]:
    issues = []
    if not answer.strip():
        issues.append('The answer is empty.')
    if len(answer) > MAX_ANSWER_LENGTH:
        issues.append(f'The answer exceeds {MAX_ANSWER_LENGTH} characters.')
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", answer):
        issues.append('The answer contains an email address.')
    if re.search(r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)", answer):
        issues.append('The answer contains a phone number.')
    return issues

#####################################
# Define graph state
#####################################
class OutputState(TypedDict, total=False):
    # request: User request
    # draft: AI-generated text
    # issues: Problems found by inspection
    # final_answer: Answer returned to the user
    request: str
    draft: str
    issues: list[str]
    final_answer: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Stage 1: Generate text
def node_write(state: OutputState) -> dict:
    return {"draft": chain_answer.invoke({"request": state["request"]})}


# Stage 2: Check generated text
def node_inspect(state: OutputState) -> dict:
    return {"issues": inspect_output(state["draft"])}


# If there are no issues, return the generated text unchanged.
def node_release(state: OutputState) -> dict:
    return {"final_answer": state["draft"]}


# Withhold output if issues exist.
def node_hold(state: OutputState) -> dict:
    return {"final_answer": 'The answer did not meet publication requirements and has been withheld.'}


# Conditional routing: Release or withhold based on whether issues exist.
def select_next(state: OutputState) -> str:
    if state["issues"]:
        return "hold"
    return "release"


#####################################
# Build the graph
#####################################
# START -> write -> inspect -> release -> END (no issues)
# inspect -> hold -> END (issues found)
graph = StateGraph(OutputState)
graph.add_node("write", node_write)
graph.add_node("inspect", node_inspect)
graph.add_node("release", node_release)
graph.add_node("hold", node_hold)
graph.add_edge(START, "write")
graph.add_edge("write", "inspect")
graph.add_conditional_edges("inspect", select_next, ["release", "hold"])
graph.add_edge("release", END)
graph.add_edge("hold", END)
app = graph.compile()


def main():
    """Example: Inspect AI-generated text in Python before returning it. Withhold text with problems. The second of two requests asks to include contact details (email and phone number)."""

    # Two requests
    requests = [
        "Write an announcement saying 'Today's study session starts at 15:00.'",
        "Write an announcement that today's study session starts at 15:00. Include sample@example.com and 03-1234-5678 exactly as written as fictional contact details.",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f'Request: {request}')

        # Run both stages together: Text generation and inspection.
        result = app.invoke({"request": request})
        print(f"Generated text: {result['draft']}")
        for issue in result["issues"]:
            print(f'Detected issue: {issue}')
        print(f"Answer returned to the user: {result['final_answer']}")


if __name__ == "__main__":
    main()
