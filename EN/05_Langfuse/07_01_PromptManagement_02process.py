import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

llm = make_llm("minimal")

#####################################
# Langfuse: Destination for execution records (traces)
#####################################
# Omitting the logging destination sends data to Langfuse Cloud. Send it to local Langfuse instead.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

# Name of the prompt registered by 07_01_PromptManagement_01preprocess.py
PROMPT_NAME = "review_extract"


#####################################
# Retrieve a prompt from Langfuse and build a chain
#####################################
def make_chain(label: str):
    # get_prompt: Retrieve a prompt by label.
    # cache_ttl_seconds: How long to cache the prompt. Zero fetches every time, immediately reflecting UI changes.
    try:
        prompt = langfuse.get_prompt(PROMPT_NAME, label=label, type="chat", cache_ttl_seconds=0)
    except Exception as error:
        raise RuntimeError(
            f"Could not retrieve prompt '{PROMPT_NAME}' (label {label}). Run 07_01_PromptManagement_01preprocess.py first."
        ) from error

    # get_langchain_prompt returns messages converting {{review}} to LangChain's {review} syntax.
    template = ChatPromptTemplate.from_messages(prompt.get_langchain_prompt())
    # Adding langfuse_prompt to metadata links the prompt version to the trace's LLM record.
    # (In Prompts, inspect usage counts, tokens, and time per version.)
    template.metadata = {"langfuse_prompt": prompt}
    return prompt, template | llm | StrOutputParser()


def main():
    """Example: Extract positives and concerns from product reviews (same as stage 1 of 01_LangChain/01_01_Chain.py). Retrieve registered Langfuse prompts instead of embedding them in code. First register them with 07_01_PromptManagement_01preprocess.py.
    Flow: The local 07_01_PromptManagement_02process.py retrieves production and staging prompts from Langfuse, processes the same review with each, and compares results.
    Try moving the production label to version 2 in Prompts -> review_extract and rerun. Without changing code, production output now uses version 2's prompt.
    """

    # Product review
    review = (
        'These wireless earbuds have clear sound. For example, surrounding noise does not bother me. '
        'I also like the small charging case, which is easy to carry. '
        'However, the connection sometimes drops when I use just one earbud. The instructions were also hard to understand.'
    )
    print(f'Review: {review}')

    for label in ["production", "staging"]:
        prompt, chain = make_chain(label)
        print()
        print(f'--- Label {label} (version {prompt.version}) ---')
        with langfuse.start_as_current_observation(name='Review organization', input={"review": review, "label": label}) as span:
            result = chain.invoke({"review": review}, config={"callbacks": [CallbackHandler()]})
            span.update(output=result)
            print(result)
            print(f'Trace: {langfuse.get_trace_url()}')

    # Records are batched in the background. Flush all remaining records before program exit.
    langfuse.flush()


if __name__ == "__main__":
    main()
