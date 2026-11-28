[Japanese source](../JP/README.md) | [English](README.md)

The Japanese version is the source of truth. This English version was translated from Japanese using an LLM.

# Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Create .env

In each folder, copy `.env.example` to `.env`.

Set API keys and other values in `.env`.

```bash
cp 00_LiteLLM/.env.example 00_LiteLLM/.env
```

## Start the database

Start PostgreSQL to run the following examples.

- `02_05_MemoryThread.py`: When postgres is selected as the storage backend
- `02_04_Retrieval_PGVector.py` (in each folder): Used for vector storage

```bash
docker compose up -d
```

The image is PostgreSQL 17 with a vector search extension (`pgvector/pgvector:pg17`).

Set `SESSION_DATABASE_URL` in `.env` as follows.

```
SESSION_DATABASE_URL=postgresql://agent_user:local_dev_password@127.0.0.1:15432/agent_sessions?sslmode=disable
```

## Start Langfuse

To run the `05_Langfuse` examples, start Langfuse locally. You do not need to obtain keys. See [05_Langfuse/README.md](05_Langfuse/README.md) for details.

```bash
docker compose -f 05_Langfuse/docker-compose.yaml up -d
```

# Obtain API keys

These examples require API keys to call LLMs.

## OpenAI

[https://platform.openai.com](https://platform.openai.com)

### Add credits

1. Select **HOME** in the left menu.
2. Select **Add Credits** to add credits.

### Create an API key

1. Select **API keys** in the left menu.
2. Select **Create new secret key** to create an API key.

## Anthropic

[https://platform.claude.com](https://platform.claude.com)

### Add credits

1. Select **Billing** in settings.
2. Select **Buy credits**, enter an amount, and purchase credits.

### Create an API key

1. Select **API keys**.
2. Select **Create key**, enter a name and other details, and create an API key.
3. Copy and save the displayed API key.
