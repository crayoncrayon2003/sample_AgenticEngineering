import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
GOAL_DIR = os.path.join(ROOT, "outputs", "goal_loop")


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


MAX_ROUNDS = 5

llm = make_llm("low")


#####################################
# Define the LLM response schema (final goal evaluation)
#####################################
class GoalReview(BaseModel):
    # feedback: Evaluation for each goal
    # goals_met: True if all goals are satisfied
    # Put evaluation before the verdict so the model explains before deciding.
    feedback: str = Field(description='Evaluation for each goal in English, including reasons for any unmet goals')
    goals_met: bool = Field(description='true only if the last executed code satisfies all goals')


#####################################
# Monitor: Save generated code and actually execute it
#####################################
# Note: Generated code is actually executed on this PC.
# These teaching safeguards are basic: run in a separate process, omit environment variables such as API keys, and stop after 10 seconds.
def save_and_run(code: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as file:
        file.write(code + "\n")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", path],
            env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return 'Execution failed: Did not finish within 10 seconds.'
    if completed.returncode != 0:
        return f'Execution failed (exit code {completed.returncode}):\n{completed.stderr.strip()[-800:]}'
    return f'Execution succeeded:\n{completed.stdout.strip()[-800:]}'


# Execution records (initialized in main)
run_log = {"dir": "", "round": 0, "last_path": "", "last_result": ""}


#####################################
# Tool: Save and execute code and return the result
#####################################
# In 04_06_GoalLoop.py, a Python for loop repeats generation, execution, and evaluation.
# Here the agent calls this tool to check execution results, then revises and calls it again if needed.
@tool
def run_code(code: str) -> str:
    """Accept a complete Python script, save it, execute it, and return the result.

    Args:
        code: Entire Python script without explanations or code fences.
    """
    run_log["round"] += 1
    path = os.path.join(run_log["dir"], f"round_{run_log['round']}.py")
    result = save_and_run(code, path)
    run_log["last_path"] = path
    run_log["last_result"] = result
    print(f"--- Attempt {run_log['round']} ---")
    print(f'Execution output: {result.splitlines()[0]}')
    return result


# system_prompt: The agent's role (both writer and reviewer)
system_prompt = (
    'You are a Python programmer. Write an entire script that meets the request and goals, and '
    'always run it using run_code to check the result.\n'
    'If execution fails or goals are unmet, revise the code and run it again with run_code.\n'
    'Use only the standard library.\n'
    'Finally check the last executed code and output against each goal and return an evaluation. '
    'If execution or an assert fails, the goals are not met.'
)
# Instructions requesting a final evaluation after tool calls finish
review_request = 'Check and evaluate the last executed code and output against each goal.'

#####################################
# Pass tools to the LLM
#####################################
# bind_tools passes the available tool list to the LLM, which returns tool calls as needed.
llm_with_tools = llm.bind_tools([run_code])
llm_review = llm.with_structured_output(GoalReview, method="json_schema")


#####################################
# Define graph state
#####################################
class AgentState(MessagesState, total=False):
    # messages: Conversation history (provided by MessagesState)
    # stopped: True if stopped by the run_code call limit
    # review: Final evaluation
    stopped: bool
    review: GoalReview


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Call the LLM. Its response is either an answer or a request to call tools.
def node_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# Execute tools requested by the LLM.
# Limit run_code calls to MAX_ROUNDS (equivalent to LangChain's ToolCallLimitMiddleware).
def node_tools(state: AgentState) -> dict:
    tool_messages = []
    for tool_call in state["messages"][-1].tool_calls:
        if run_log["round"] >= MAX_ROUNDS:
            print(f'  [Limit] Stopping the agent: run_code is limited to {MAX_ROUNDS} calls.')
            return {"messages": tool_messages, "stopped": True}
        tool_messages.append(run_code.invoke(tool_call))
    return {"messages": tool_messages}


# Request the final evaluation as GoalReview (equivalent to LangChain's response_format).
def node_review(state: AgentState) -> dict:
    review = llm_review.invoke([SystemMessage(content=system_prompt)] + state["messages"] + [HumanMessage(content=review_request)])
    return {"review": review}


# Conditional routing 1: Go to tools if calls exist; otherwise go to final evaluation.
def select_after_model(state: AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "review"


# Conditional routing 2: End if stopped by the call limit; otherwise return to the LLM.
def select_after_tools(state: AgentState) -> str:
    if state.get("stopped"):
        return END
    return "model"


#####################################
# Agent (manually connect an LLM node and a tool-execution node in a loop)
#####################################
# START -> model -> review -> END (no tool calls)
#            ↑    │
# Loop through tools when tool calls are present.
# tools -> END (run_code call limit)
graph = StateGraph(AgentState)
graph.add_node("model", node_model)
graph.add_node("tools", node_tools)
graph.add_node("review", node_review)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", select_after_model, ["tools", "review"])
graph.add_conditional_edges("tools", select_after_tools, ["model", END])
graph.add_edge("review", END)
app = graph.compile()

# recursion_limit: Maximum graph steps (node executions), a LangGraph feature
# Exceeding the limit raises GraphRecursionError. Use it to bound LLM calls (equivalent to LangChain's ModelCallLimitMiddleware).
# Each iteration uses two steps (model and tools); allow MAX_ROUNDS iterations plus the final model and review steps.
RECURSION_LIMIT = MAX_ROUNDS * 2 + 2


# Run the graph and return the final evaluation, or None if a limit is reached.
def run_agent(request: str) -> GoalReview | None:
    try:
        result = app.invoke({"messages": [{"role": "user", "content": request}]}, {"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError:
        print(f'  [Limit] Stopping the agent after reaching {RECURSION_LIMIT} steps.')
        return None
    return result.get("review")


def main():
    """Example: Generate a Python function using the request and goals of 04_06_GoalLoop.py and the flow of 01_LangChain/04_06_GoalLoop_Builtin.py.
    Difference: 04_06_GoalLoop.py uses a Python loop for writer (LLM) -> execution -> reviewer (LLM), up to five rounds. Here the agent executes via run_code, checks output, and revises. Limits on run_code and LLM calls bound the loop; the agent returns a final GoalReview.
    Flow: 1. Write and run code with run_code. 2. Check results and revise/re-execute if needed (up to five times). 3. Return a final evaluation per goal (GoalReview). 4. Python accepts the result only if the last execution succeeded and all goals were judged met. Save each round's code in outputs/goal_loop/.
    """

    use_case = (
        'Create binary_gap(n), which accepts a positive integer and returns the maximum number of consecutive zeros enclosed by ones in its binary representation. '
        'Return zero if there is no such run of zeros.'
    )
    goals = [
        'Accept a positive integer and return the result according to the definition.',
        'Raise TypeError for non-integers (including bool) and ValueError for integers of zero or less.',
        'Include asserts checking 9->2, 529->4, 20->1, 15->0, 32->0, and 1->0.',
        "Display usage examples and results inside if __name__ == '__main__':.",
    ]
    print('Goals:')
    for goal in goals:
        print(f'  - {goal}')

    # Directory storing each round's code
    run_log.update({
        "dir": os.path.join(GOAL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + "_builtin"),
        "round": 0, "last_path": "", "last_result": "",
    })
    os.makedirs(run_log["dir"], exist_ok=True)

    goals_text = "\n".join(f"- {goal}" for goal in goals)
    request = f'[Request]\n{use_case}\n\n[Goals]\n{goals_text}'
    review = run_agent(request)

    # If a limit stops execution, there is no final evaluation.
    if review is None:
        print(f"Stopped early at the call limit. Records: {run_log['dir']}")
        return
    print(f"Evaluation: {'Goal achieved' if review.goals_met else 'Not achieved'}")
    print(f"  {review.feedback}")

    if review.goals_met and run_log["last_result"].startswith('Execution succeeded'):
        print(f"All goals met. Accepted code: {run_log['last_path']}")
    else:
        print(f"Goal not achieved. Records: {run_log['dir']}")


if __name__ == "__main__":
    main()
