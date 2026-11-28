# 10_Mem0

Examples of mem0, a library that extracts facts about users from conversations and stores and searches them as long-term memories.

The long-term memory parts of the 01_LangChain examples are replaced with mem0.

Compare files with the same number in 01_LangChain and 10_Mem0 to see the differences.

## Setup

```bash
cp 10_Mem0/.env.example 10_Mem0/.env
```

| Variable | Required | Description |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |

mem0 usage telemetry is disabled by setting `MEM0_TELEMETRY=False`.

## LLM used for fact extraction

By default, mem0 calls OpenAI directly to extract facts from messages.

02_07_MemoryUserProfile.py instead passes a LangChain model created by `make_llm` through mem0's `llm` configuration with `provider: langchain`.

This allows switching between OpenAI and Claude with LLM_PROVIDER.

Embeddings for memory search use OpenAI (`text-embedding-3-small`).

## Notes

- Run with `LLM_PROVIDER=openai` because this example's embeddings require OpenAI (as in 01_LangChain/02_07).
- Memories are stored in Qdrant in memory only while the program runs (as with `InMemoryStore` in 01_LangChain/02_07).
- mem0 configuration (`config.json`) and memory change history (`history.db`) are stored in `10_Mem0/outputs/mem0/`, rather than the default home folder (`~/.mem0`).
- `custom_instructions` specifies how to extract facts (English sentences in the form "The user ...", and no extraction from questions alone).
