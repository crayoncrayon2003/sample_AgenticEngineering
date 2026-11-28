import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
DOCS_DIR = os.path.join(ROOT, "data", "docs")
COLLECTION_NAME = "sample_docs"  # Name identifying a document collection within the vector store

# Embeddings are only available from OpenAI in this example, so Claude cannot run it.
if PROVIDER != "openai":
    sys.exit('This example uses embeddings (OpenAI only). Run with LLM_PROVIDER=openai.')


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
# Load documents and save them to a vector store (PostgreSQL + pgvector).
#####################################
# Differences from 02_04_Retrieval.py:
# InMemoryVectorStore disappears when the program ends, while PGVector persists in the database.
# Stable chunk IDs (filename-number) make repeated runs overwrite existing records instead of duplicating them.
def build_vector_store() -> PGVector:
    # Save using SESSION_DATABASE_URL in .env (run docker compose up -d first).
    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit('Set SESSION_DATABASE_URL in .env.')
    # PGVector connects through SQLAlchemy, so convert the URL to postgresql+psycopg:// format.
    connection = url.replace("postgresql://", "postgresql+psycopg://", 1)

    store = PGVector(
        # embeddings: How text is converted to vectors
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        # collection_name: Name of the document collection
        collection_name=COLLECTION_NAME,
        # connection: Destination database
        connection=connection,
        # use_jsonb: Store metadata such as sources as JSONB
        use_jsonb=True,
    )

    # Read each .txt file in data/docs.
    documents = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            documents.append(Document(page_content=file.read(), metadata={"source": filename}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    # Chunk IDs: Assign the same ID to the same position in the same file on every run.
    ids = []
    counts = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counts[source] = counts.get(source, 0) + 1
        ids.append(f"{source}-{counts[source]}")

    store.add_documents(chunks, ids=ids)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks and saved them to PostgreSQL.')
    return store


vector_store = build_vector_store()


#####################################
# Search: Find internal-document passages related to the question
#####################################
def search_documents(query: str) -> str:
    """Search internal documents for passages related to the question."""
    results = vector_store.similarity_search_with_score(query, k=3)
    print(f'  [Search] {query}')
    lines = []
    for doc, score in results:
        # PGVector scores are distances: Smaller means closer, unlike InMemoryVectorStore similarity.
        print(f"    - {doc.metadata['source']} (distance {score:.2f})")
        lines.append(f"[Source: {doc.metadata['source']}]\n{doc.page_content}")
    return "\n\n".join(lines) or 'No matching documents were found.'


#####################################
# Answer generation chain
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You are an internal-document guide. '
        'Use only information in the search results and cite the filenames used as evidence. '
        'If the results do not confirm the answer, say "I could not confirm it in the documents". Do not fill gaps with guesses.',
    ),(
        # user: A message from the application user
        "user",
        '[Question]\n{question}\n\n[Search results]\n{context}'
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# Define graph state
#####################################
class RagState(TypedDict, total=False):
    # question: Question
    # context: Search results
    # answer: Answer
    question: str
    context: str
    answer: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Stage 1: Search for documents related to the question
def node_retrieve(state: RagState) -> dict:
    return {"context": search_documents(state["question"])}


# Stage 2: Answer using only search results
def node_generate(state: RagState) -> dict:
    return {"answer": chain_answer.invoke({"question": state["question"], "context": state["context"]})}


#####################################
# Build the graph
#####################################
#  START → retrieve → generate → END
graph = StateGraph(RagState)
graph.add_node("retrieve", node_retrieve)
graph.add_node("generate", node_generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()


def main():
    """Example: Answer the same four questions as 02_04_Retrieval.py about fictional internal documents (data/docs); the last is unanswerable. The program fixes search -> answer ordering. The difference is vector storage in PostgreSQL (pgvector)."""

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
        # Run both stages together: Search, then answer.
        result = app.invoke({"question": question})
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    main()
