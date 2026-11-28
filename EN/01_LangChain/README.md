# 01_LangChain

Examples of software using LLMs, written with LangChain.

## Setup

Copy `.env.example` to create `01_LangChain/.env`, then set its values.

```bash
cp 01_LangChain/.env.example 01_LangChain/.env
```

| Variable | Required | Description |
| ------------------------ | :--------------: | --------------------------------------------------------------------- |
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `OPENAI_VECTOR_STORE_ID` | | Vector store ID for file_search in 02_03 |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |
| `SESSION_DATABASE_URL` | | Connection URL for PostgreSQL in 02_05 and 02_04_Retrieval_PGVector |

Empty variable values are treated the same as omitted variables.

## Switch between OpenAI and Claude

Use `LLM_PROVIDER` in `.env` to switch between OpenAI and Claude.
Some examples contain code for only OpenAI or only Claude.

## Middleware execution order

Middleware passed to `create_agent(middleware=[...])` is called inside `agent.invoke()` in the following order.
When tools are used, the LLM call -> tool execution cycle repeats until no tools are needed.

```
agent.invoke() starts
  ↓
@before_agent: Once at the start
  ↓
@before_model
  ↓
LLM call (@dynamic_prompt and @wrap_model_call wrap this call)
  ↓
@after_model
  ↓
Are tool calls needed?
  ├─ Yes
  │    ↓
  │  @wrap_tool_call: Before execution
  │    ↓
  │  handler(request): Execute the tool
  │    ↓
  │  @wrap_tool_call: After execution and exception handling
  │    ↓
  │  Return to @before_model
  │
  └─ No: Final answer
       ↓
@after_agent: Once on successful completion
  ↓
agent.invoke() returns the result
```

| Decorator | When called | Main uses |
| -------------------- | -------------------------------- | -------------------------------------------- |
| `@before_agent` | Once at the start of `invoke()` | Input checks and preparation |
| `@before_model` | Before each LLM call | Manage conversation history and add instructions |
| `@dynamic_prompt` | On each LLM call | Create a system prompt based on the situation |
| `@wrap_model_call` | Wraps each LLM call | Processing before/after calls, retries, and model switching |
| `@after_model` | After each LLM response | Check responses |
| `@wrap_tool_call` | Wraps each tool execution | Validate arguments and handle exceptions |
| `@after_agent` | Once after the final answer | Save results and perform cleanup |
