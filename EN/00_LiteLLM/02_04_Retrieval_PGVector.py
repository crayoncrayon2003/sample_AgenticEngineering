import json
import os
import sys

import litellm
import psycopg
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
DOCS_DIR = os.path.join(ROOT, "data", "docs")
TABLE_NAME = "doc_chunks"  # Table storing chunks and vectors

# Embeddings are only available from OpenAI in this example, so Claude cannot run it.
if PROVIDER != "openai":
    sys.exit('This example uses embeddings (OpenAI only). Run with LLM_PROVIDER=openai.')


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
# Load documents and create embeddings
#####################################
# Split text into chunks of about 120 characters (a simplified equivalent of LangChain's RecursiveCharacterTextSplitter).
def split_text(text: str, chunk_size: int = 120) -> list[str]:
    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) > chunk_size:
            chunks.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# Convert text to embedding vectors (sequences of numbers).
def embed(texts: list[str]) -> list[list[float]]:
    response = litellm.embedding(model="openai/text-embedding-3-small", input=texts)
    return [item["embedding"] for item in response.data]


# Convert vectors to pgvector-compatible strings (e.g., [0.1,0.2,0.3]).
def to_vector_text(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


# Connect to SESSION_DATABASE_URL in .env (run docker compose up -d first).
def connect() -> psycopg.Connection:
    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit('Set SESSION_DATABASE_URL in .env.')
    return psycopg.connect(url, autocommit=True)


# Differences from 02_04_Retrieval.py:
# 02_04_Retrieval.py keeps vectors in a Python list, so they disappear when the program ends.
# Here we store them in a PostgreSQL table (equivalent to LangChain's PGVector).
# Assign a stable ID (filename-number) to each chunk and overwrite existing IDs to prevent duplicates on repeated runs.
def build_vector_store() -> None:
    # Read each .txt file in data/docs and split it into chunks.
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            for number, text in enumerate(split_text(file.read()), start=1):
                chunks.append({"id": f"{filename}-{number}", "source": filename, "text": text})

    vectors = embed([chunk["text"] for chunk in chunks])
    with connect() as conn:
        # Enable pgvector and create a table with a vector column (1536 dimensions, the size of text-embedding-3-small).
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} "
            "(id text PRIMARY KEY, source text NOT NULL, text text NOT NULL, embedding vector(1536) NOT NULL)"
        )
        # Overwrite an existing ID (ON CONFLICT ... DO UPDATE).
        for chunk, vector in zip(chunks, vectors):
            conn.execute(
                f"INSERT INTO {TABLE_NAME} (id, source, text, embedding) VALUES (%s, %s, %s, %s::vector) "
                "ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source, text = EXCLUDED.text, embedding = EXCLUDED.embedding",
                (chunk["id"], chunk["source"], chunk["text"], to_vector_text(vector)),
            )
    print(f'Split into {len(chunks)} chunks and saved to PostgreSQL.')


build_vector_store()


#####################################
# Search tool
#####################################
def search_documents(query: str) -> str:
    """Search internal documents for passages related to the question."""
    query_vector = embed([query])[0]
    # <=>: pgvector cosine distance. Smaller means closer (the reverse of the similarity in 02_04_Retrieval.py).
    with connect() as conn:
        rows = conn.execute(
            f"SELECT source, text, embedding <=> %s::vector AS distance FROM {TABLE_NAME} ORDER BY distance LIMIT 3",
            (to_vector_text(query_vector),),
        ).fetchall()
    print(f'  [Search] {query}')
    lines = []
    for source, text, distance in rows:
        print(f'    - {source} (distance {distance:.2f})')
        lines.append(f'[Source: {source}]\n{text}')
    return "\n\n".join(lines) or 'No matching documents were found.'


# Tool descriptions sent to the LLM
tools = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": 'Search internal documents for passages related to the question.',
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": 'A search query describing what to look up'}},
            "required": ["query"],
        },
    }},
]


#####################################
# Agent
#####################################
# system_prompt: The agent's role
system_prompt = (
    'You are an internal-document guide. Always use search_documents before answering. '
    'Use only information in the search results and cite the filenames used as evidence. '
    'If the results do not confirm the answer, say "I could not confirm it in the documents". Do not fill gaps with guesses.'
)


def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    message = call_llm(llm, messages, tools=tools)

    # While the LLM requests tool calls, execute the tools and call the LLM again.
    while message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = search_documents(args["query"])
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        message = call_llm(llm, messages, tools=tools)

    return message.content


def main():
    """Example: Answer four questions about fictional internal documents (data/docs); the last is unanswerable. Split documents into chunks, embed them, then embed questions and find nearby passages using PostgreSQL (pgvector). Unlike 02_04_Retrieval.py, PostgreSQL handles vector storage and search."""

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
        # Run the agent
        answer = run_agent(question)
        print(f'Answer: {answer}')


if __name__ == "__main__":
    main()
