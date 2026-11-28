import os
import sys

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")


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


# Use built-in tools through LiteLLM's responses interface (same format as OpenAI's Responses API).
# This example uses OpenAI's built-in tools and is OpenAI-only.
if PROVIDER != "openai":
    sys.exit("This example uses OpenAI's built-in tools. Run with LLM_PROVIDER=openai.")

llm = make_llm("low")

#####################################
# Built-in tools
#####################################
# Specify tools as dictionaries rather than functions. The provider's server performs the actual work.
tool_web_search = {"type": "web_search"}
tool_code_execution = {"type": "code_interpreter", "container": {"type": "auto"}}


#####################################
# Ask a question and display the answer
#####################################
# Ask using built-in tools, then display and return the answer
def ask(tool: dict, instructions: str, question: str):
    print(f'Question: {question}')
    response = litellm.responses(
        model=llm["model"],
        reasoning={"effort": llm["reasoning_effort"]},
        instructions=instructions,
        input=question,
        tools=[tool],
        timeout=180,
    )
    # output contains tool execution records followed by the final answer (message).
    answer = response.output[-1]
    print(f'Answer: {answer.content[0].text}')
    return response


# Display answer citations (URLs or filenames).
def show_citations(response) -> None:
    for citation in response.output[-1].content[0].annotations or []:
        print(f"  Source: {getattr(citation, 'title', '') or getattr(citation, 'filename', '')} {getattr(citation, 'url', '')}")


# Display code executed on the provider's server.
def show_executed_code(response) -> None:
    for item in response.output:
        if item.type == "code_interpreter_call":
            print('  [Executed code]')
            print(item.code)


def main():
    """Example: Three uses of provider-hosted built-in tools (OpenAI only): 1. Web search: Answer recent questions with citations. 2. Code execution: Run Python on the provider's server. 3. File search: Answer by searching documents registered with OpenAI."""

    print('========== 1. Web search ==========')
    response = ask(tool_web_search, 'Always use web search, prioritize primary sources, and include dates and citations.',
                   'What is the latest stable Python version and its release date?')
    show_citations(response)

    print('========== 2. Code execution ==========')
    response = ask(tool_code_execution, 'Always use the code execution tool to run Python for calculations and answer briefly.',
                   'Calculate the sum of the digits of 30 factorial.')
    show_executed_code(response)

    print('========== 3. File search ==========')
    if not VECTOR_STORE_ID:
        print('Cannot run because OPENAI_VECTOR_STORE_ID is missing from .env.')
        return
    tool_file_search = {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}
    # The second question has no answer in the documents.
    for question in ['What is the second-quarter sales target?', "What is Laboratory X's phone number?"]:
        response = ask(tool_file_search, 'Always look up the answer with file_search. If it is not found, answer "I could not confirm it in the search results".',
                       question)
        show_citations(response)


if __name__ == "__main__":
    main()
