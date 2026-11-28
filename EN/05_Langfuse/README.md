# 05_Langfuse

The same examples as 01_LangChain, with Langfuse tracing (execution records), evaluation, and prompt management added.

Langfuse is open source and can run in Docker on your PC. No key acquisition or account registration is required.

Compare files with the same number in 01_LangChain and 05_Langfuse to see the parts added for Langfuse.

## Setup

### 1. Start Langfuse

Run the following. Downloading images may take several minutes on the first run.

```bash
docker compose docker-compose.yaml up
```

Six containers start: GUI, processing, PostgreSQL, ClickHouse, Redis, and MinIO.

They use several GB of memory.

Once running, open [http://localhost:3000](http://localhost:3000) in your browser.

Log in with the following credentials.

| Item | Value |
| ---- | -- |
| Email address | `admin@example.com` |
| Password | `local_dev_password` |

Select "Projects" in the left menu, then "Go to project" for "Agentic Engineering Sample".

Select "Tracing" in the left menu, find the relevant log, and click it to inspect the contents.

When finished, stop it with the following command. Recorded data remain in Docker volumes and are available on the next start.

```bash
docker compose -f 05_Langfuse/docker-compose.yaml down
```

To delete the data too, use `down -v`.

### 2. Create .env

```bash
cp 05_Langfuse/.env.example 05_Langfuse/.env
```

| Variable | Required | Description |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |
| `LANGFUSE_BASE_URL` | | Langfuse endpoint. Defaults to `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse public key. Keep the value from `.env.example` (the key created in `docker-compose.yaml`) |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse secret key. Keep the value from `.env.example` |

If `LANGFUSE_BASE_URL` is omitted, the Langfuse SDK attempts to send records to Langfuse Cloud on the internet.
These examples default to `http://localhost:3000`, so records are sent only to local Langfuse.

## Add tracing

### Required pattern

These two components record LangChain execution.

| Component | Role |
| ---- | ---- |
| `CallbackHandler()` | Records LangChain execution and sends it to Langfuse |
| `langfuse.flush()` | Unsent records are lost when the program exits. Send all remaining records at the end |

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

result = agent.invoke(
    { "messages": [...] },
    config = {"callbacks": [CallbackHandler()]}
)

langfuse.flush()
```


### Optional pattern

The required pattern alone is enough to record traces.

Each `invoke` call creates a separate trace with an automatically assigned name.

The following components can group traces and add names or tags, making logs easier to find.

| Component | Role |
| ---- | ---- |
| `langfuse.start_as_current_observation(name=...)` | Groups multiple executions into one trace. Can also record trace input and output |
| `langfuse_session_id` in `config.metadata` | Groups traces from the same conversation into one session (02_05) |
| `langfuse_tags` in `config.metadata` | Adds tags to traces (03_07) |
| `propagate_attributes(...)` | Adds names, tags, session IDs, and metadata to traces |
| `langfuse.get_trace_url()` | Returns a URL to open the current trace in the GUI |

### Features outside the scope of these examples
#### Cost display

Langfuse can display LLM call costs per trace and per model.

These examples omit the steps for registering models and prices in the GUI.
Register models and prices in Settings -> Models to display costs during analysis.

#### Execution within Langfuse

Langfuse can call LLMs directly to try and evaluate prompts.

In these examples, all LLM calls and grading run on the local PC; Langfuse stores and displays results.

Configure LLM details (URL, API key, etc.) in Settings to use the following Langfuse features.

**Automatic LLM evaluation (LLM-as-a-Judge)**

 Use another LLM to assess an LLM's answer to a question and record the rating as a score.

 This can be configured in the Langfuse GUI without writing a program.

**Playground**

Edit prompts in the GUI and try the LLM's answers.


#### Dashboards
Visualize application usage and quality changes as charts.

Select a time range to inspect trends in trace counts, time, token usage, costs, and scores.

Annotation Queues also let people review and grade answers and save those results as scores.

Both are operated directly in the GUI; these examples contain no code for them.
