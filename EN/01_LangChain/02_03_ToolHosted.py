import os

from dotenv import load_dotenv
from langchain.agents import create_agent

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")


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
            timeout=180,
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
            timeout=180,
            max_retries=1,
        )
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


llm = make_llm("low")

#####################################
# Built-in tools from the LLM provider (OpenAI/Anthropic), with provider-specific syntax
#####################################
# Specify built-in tools as dictionaries. Tools run on the provider's server.
if PROVIDER == "openai":
    tool_web_search = {"type": "web_search"}
    tool_code_execution = {"type": "code_interpreter", "container": {"type": "auto"}}
    tool_file_search = {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}
else:
    tool_web_search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
    tool_code_execution = {"type": "code_execution_20260521", "name": "code_execution"}
    tool_file_search = None

#####################################
# Agent
#####################################
# Web-search agent using tool_web_search
agent_web = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[tool_web_search],
    # system_prompt: The agent's role
    system_prompt='Always use web search, prioritize primary sources, and include dates and citations.',
)

# Code-execution agent using tool_code_execution
agent_code = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[tool_code_execution],
    # system_prompt: The agent's role
    system_prompt='Always use the code execution tool to run Python for calculations and answer briefly.',
)
# File-search agent, OpenAI only, using tool_code_execution (as stated in the original comment)
if PROVIDER == "openai" and VECTOR_STORE_ID:
    agent_file = create_agent(
        model=llm,
        tools=[tool_file_search],
        system_prompt='Always look up the answer with file_search. If it is not found, answer "I could not confirm it in the search results".',
    )
else:
    agent_file = None


#####################################
# Ask a question and display the answer
#####################################
# Ask the agent, then display and return its answer.
def ask(agent, question: str):
    print(f'Question: {question}')
    # Run the agent ("user": A message from the application user).
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1]
    print(f'Answer: {answer.text}')
    return answer


#####################################
# Display results
#####################################
# Display web-search/file-search agent responses
def show_citations(answer) -> None:
    for block in answer.content_blocks:
        for citation in block.get("annotations", []):
            print(f"  Source: {citation.get('title', '')} {citation.get('url', '')}")


# Display code-execution agent responses
def show_executed_code(answer) -> None:
    for block in answer.content_blocks:
        if block.get("type") == "server_tool_call":
            print('  [Executed code]')
            print(block.get("args", {}).get("code", ""))


def main():
    """Example: Three agents using provider-hosted tools. 1. Web search: Answer recent questions with citations. 2. Code execution: Calculate using Python on the provider's server. 3. File search: Search documents registered with OpenAI (OpenAI only)."""

    print('========== 1. Web search ==========')
    answer = ask(agent_web, 'What is the latest stable Python version and its release date?')
    show_citations(answer)

    print('========== 2. Code execution ==========')
    answer = ask(agent_code, 'Calculate the sum of the digits of 30 factorial.')
    show_executed_code(answer)

    print('========== 3. File search ==========')
    if agent_file is None:
        print('Cannot run: Either the provider is not OpenAI or OPENAI_VECTOR_STORE_ID is missing from .env.')
        return

    for question in ['What is the second-quarter sales target?', "What is Laboratory X's phone number?"]:
        answer = ask(agent_file, question)
        show_citations(answer)


if __name__ == "__main__":
    main()
