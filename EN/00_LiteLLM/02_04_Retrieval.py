import json
import math
import os
import sys

import litellm
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
DOCS_DIR = os.path.join(ROOT, "data", "docs")

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


def build_vector_store() -> list[dict]:
    # Read each .txt file in data/docs and split it into chunks.
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            for text in split_text(file.read()):
                chunks.append({"source": filename, "text": text})
    print(f'Split into {len(chunks)} chunks.')

    # Embed each chunk and store both together (equivalent to LangChain's InMemoryVectorStore).
    vectors = embed([chunk["text"] for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks


vector_store = build_vector_store()


# Calculate how close two vectors are using cosine similarity.
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


#####################################
# Search tool
#####################################
def search_documents(query: str) -> str:
    """Search internal documents for passages related to the question."""
    query_vector = embed([query])[0]
    scored = sorted(vector_store, key=lambda chunk: cosine_similarity(query_vector, chunk["vector"]), reverse=True)
    print(f'  [Search] {query}')
    lines = []
    for chunk in scored[:3]:
        score = cosine_similarity(query_vector, chunk["vector"])
        print(f"    - {chunk['source']} (similarity {score:.2f})")
        lines.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
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
    """Example: Answer four questions about fictional internal documents (data/docs); the last has no answer in the documents. 1. Split documents into chunks. 2. Embed the chunks. 3. Embed the question and find similar passages."""

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
