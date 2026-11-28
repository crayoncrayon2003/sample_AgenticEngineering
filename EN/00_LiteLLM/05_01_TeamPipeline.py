import os

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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


llm = make_llm("minimal")

#####################################
# Agent 1: Researcher
#####################################
# system_prompt: The agent's role
system_prompt_researcher = (
    'You research and organize AI topics. Summarize major trends, practical uses, impacts, and caveats in English. '
    'No search tool is available; answer within your existing knowledge. '
    'Clearly identify uncertainty and do not invent unverified sources or URLs.'
)


# Ask the researcher and return the answer (one LLM call because there are no tools).
def agent_researcher(message: str) -> str:
    messages = [{"role": "system", "content": system_prompt_researcher}, {"role": "user", "content": message}]
    return call_llm(llm, messages).content


#####################################
# Agent 2: Writer
#####################################
# system_prompt: The agent's role
system_prompt_writer = (
    'You write technical articles. Use the supplied research summary to write a English blog post for general readers. '
    'Include a title and headings and use about 200 characters. '
    'Do not assert uncertain findings as facts or add unsupported numbers or sources.'
)


# Ask the writer and return the answer (one LLM call because there are no tools).
def agent_writer(message: str) -> str:
    messages = [{"role": "system", "content": system_prompt_writer}, {"role": "user", "content": message}]
    return call_llm(llm, messages).content


def main():
    """Example: Two agents hand off work in sequence. The researcher organizes major trends; the writer turns those findings into a blog post."""

    topic = 'Major trends in AI agents'
    print(f'Topic: {topic}')

    # Stage 1: The researcher organizes the topic
    findings = agent_researcher(f"Topic: {topic}\nSelect three major trends and describe each trend's key points, practical uses, impacts, and caveats.")
    print('--- Stage 1: Research findings ---')
    print(findings)

    # Stage 2: The writer receives the findings and writes an article
    article = agent_writer(f'Topic: {topic}\nWrite a blog post based on these findings.\n\n[Findings]\n{findings}')
    print("--- Stage 2: Writer's article ---")
    print(article)


if __name__ == "__main__":
    main()
