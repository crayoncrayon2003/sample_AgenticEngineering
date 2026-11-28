# 00_LiteLLM

Examples of software using LLMs, written with LangChain.

## Setup

| Variable | Required | Description |
|---|:-:|---|
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `OPENAI_VECTOR_STORE_ID` | | Vector store ID for file search in 02_03 |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |
| `SESSION_DATABASE_URL` | | Connection URL for PostgreSQL in 02_05 and 02_04_Retrieval_PGVector |
