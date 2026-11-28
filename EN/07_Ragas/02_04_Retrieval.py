import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Disable Ragas telemetry before importing Ragas.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance, Faithfulness

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic

# Embeddings are only available from OpenAI in this example, so Claude cannot run it.
if PROVIDER != "openai":
    sys.exit('This example uses embeddings (OpenAI only). Run with LLM_PROVIDER=openai.')

DOCS_DIR = os.path.join(ROOT, "data", "docs")


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
# LLM used for Ragas evaluation
#####################################
# Ragas metrics use a Ragas-specific LLM from llm_factory, rather than a LangChain model. Create one based on LLM_PROVIDER.
# max_tokens: Reasoning models also spend tokens on reasoning, so allow extra tokens to avoid interrupting evaluation.
def make_ragas_llm():
    if PROVIDER == "openai":
        from openai import OpenAI

        return llm_factory(os.getenv("OPENAI_MODEL") or "gpt-5-nano", client=OpenAI(), max_tokens=16000)
    from anthropic import Anthropic

    return llm_factory(os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5", provider="anthropic", client=Anthropic(), max_tokens=16000)


llm = make_llm("low")


#####################################
# Load documents and create a vector store
#####################################
def build_vector_store() -> InMemoryVectorStore:
    # Read each .txt file in data/docs.
    documents = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            documents.append(Document(page_content=file.read(), metadata={"source": filename}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')

    store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents(chunks)
    return store


vector_store = build_vector_store()

# Record retrieved documents for evaluation (clear before each question).
retrieved_contexts = []


#####################################
# Search tool
#####################################
@tool
def search_documents(query: str) -> str:
    """Search internal documents for passages related to the question.

    Args:
        query: A search query describing what to look up.
    """
    results = vector_store.similarity_search_with_score(query, k=3)
    print(f'  [Search] {query}')
    lines = []
    for doc, score in results:
        print(f"    - {doc.metadata['source']} (similarity {score:.2f})")
        lines.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content}")
        retrieved_contexts.append(doc.page_content)
    return "\n\n".join(lines) or 'No matching documents were found.'


#####################################
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[search_documents],
    # system_prompt: The agent's role
    system_prompt=(
        'You are an internal-document guide. Always use search_documents before answering. '
        'Use only information in the search results and cite the filenames used as evidence. '
        'If the results do not confirm the answer, say "I could not confirm it in the documents". Do not fill gaps with guesses.'
    ),
)


#####################################
# Evaluation: Score answers and search results with Ragas metrics
#####################################
# Faithfulness: Whether the answer is based only on search results (0-1; 1 means all claims are supported).
# ContextRelevance: Whether search results relate to the question (0-1).
ragas_llm = make_ragas_llm()
metric_faithfulness = Faithfulness(llm=ragas_llm)
metric_relevance = ContextRelevance(llm=ragas_llm)


def main():
    """Example: Evaluate an internal-document RAG agent with Ragas. Answers are text without a single correct output, so use LLM grading. The agent searches data/docs using search_documents and answers only from results; embeddings find semantically similar documents.
    Four questions cover second-quarter sales targets and priority regions, Laboratory X chemical-handling staffing and hours, sales/safety contacts, and Laboratory X's phone number (absent from the documents).
    For each question, grade Faithfulness (supported by results without guesses) and ContextRelevance (retrieval relevance) from 0 (not met) to 1 (fully met). Use make_ragas_llm rather than a LangChain model. Display both scores without pass/fail.
    Flow: Search and answer, saving documents in retrieved_contexts; pass question, answer, and contexts to metrics; display results. Only evaluation is added to 01_LangChain/02_04_Retrieval.py; agent, search, and questions are unchanged. Run with LLM_PROVIDER=openai because this example's embeddings require OpenAI.
    """

    # Four questions
    questions = [
        'What are the second-quarter sales target and priority regions?',
        'How many people must be present when handling chemicals in Laboratory X, and what are its opening hours?',
        'Who is responsible for sales initiatives and safety management, respectively?',
        "Tell me Laboratory X's phone number.",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f'Question: {question}')
        # Run the agent ("user": A message from the application user).
        retrieved_contexts.clear()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"Answer: {result['messages'][-1].text}")

        # Evaluate the answer and retrieved documents.
        faithfulness = metric_faithfulness.score(user_input=question, response=result["messages"][-1].text, retrieved_contexts=retrieved_contexts)
        relevance = metric_relevance.score(user_input=question, retrieved_contexts=retrieved_contexts)
        print(f'Evaluation: Answer faithfulness to search results {faithfulness.value:.2f}; relevance of results to the question {relevance.value:.2f}')


if __name__ == "__main__":
    main()
