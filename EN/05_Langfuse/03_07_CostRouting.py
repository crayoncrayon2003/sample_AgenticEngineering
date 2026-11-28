import os
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse import get_client
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


#####################################
# Langfuse: Destination for execution records (traces)
#####################################
# Omitting the logging destination sends data to Langfuse Cloud. Send it to local Langfuse instead.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

#####################################
# Two models: A lightweight model and a more capable model
#####################################
# Prices and speeds vary by provider and time. Check each provider's pricing page for current information.
if PROVIDER == "openai":
    MODEL_NAMES = {"light": "gpt-5-nano", "advanced": "gpt-5-mini"}
else:
    MODEL_NAMES = {"light": "claude-haiku-5-5", "advanced": "claude-sonnet-5-5"}

# Routing criterion: Question length (a simple teaching heuristic, not difficulty itself).
CHARACTER_THRESHOLD = 100


# Create an LLM with the specified model name (same settings as make_llm, changing only the model name).
def make_llm_with_model(model_name: str):
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, reasoning_effort="minimal", use_responses_api=True, timeout=60, max_retries=1)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_name, reasoning_effort="low", max_tokens=16000, timeout=60, max_retries=1)


make_llm("minimal")  # Only check the API key.

#####################################
# Agents: Create one per model
#####################################
system_prompt = (
    'You explain AI agents and data processing. '
    'Answer clearly in English. Do not assert guesses about unknown specifications as facts.'
)
agent_light = create_agent(model=make_llm_with_model(MODEL_NAMES["light"]), tools=[], system_prompt=system_prompt)
agent_advanced = create_agent(model=make_llm_with_model(MODEL_NAMES["advanced"]), tools=[], system_prompt=system_prompt)


# Select an agent based on question length.
def select_route(question: str) -> str:
    if len(question) < CHARACTER_THRESHOLD:
        return "light"
    return "advanced"


def main():
    """Example: Route short and long questions to lightweight and more capable models to reduce cost and time. Two questions demonstrate the difference."""

    print(f'Routing criterion: light below {CHARACTER_THRESHOLD} characters; advanced otherwise')

    # Two questions
    questions = [
        'What is ETL?',
        'I want to load multiple CSV files daily, handle different column names and encodings, missing values and duplicates, and save aggregated results as JSON. Account for partial file-read failures. Explain processing order, validation at each stage, retry and resume methods, and design considerations.',
    ]
    for question in questions:
        print("----------------------------------------")
        print(f'Question ({len(question)} characters): {question}')

        # Stage 1: Select a model
        route = select_route(question)
        print(f'Selected route: {route} ({MODEL_NAMES[route]})')

        # Stage 2: Answer with the selected model's agent and record time and usage
        if route == "light":
            agent = agent_light
        else:
            agent = agent_advanced
        start = time.perf_counter()
        # Langfuse: Record routing destinations as tags (compare tokens and time by tag in the UI).
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_tags": [route]}})
        elapsed = time.perf_counter() - start

        answer = result["messages"][-1]
        usage = answer.usage_metadata or {}
        print(f"Time: {elapsed:.1f} seconds; tokens: input {usage.get('input_tokens')}, output {usage.get('output_tokens')}")
        print(f'Answer: {answer.text}')

    # Langfuse: Records are batched in the background; flush all remaining records before program exit.
    langfuse.flush()


if __name__ == "__main__":
    main()
