# 02_LangGraph

The same examples as 01_LangChain, rewritten with LangGraph.

Prompts and chains are the same as in 01_LangChain.

Only the way the control flow is written differs.

Compare files with the same number in 01_LangChain and 02_LangGraph to see the differences.

## Functional API concepts

| Component | Role |
|---|---|
| `@task` | One workflow step. Calling it immediately returns a future; retrieve the result with `.result()`. Calling another task before `.result()` allows concurrent execution |
| `@entrypoint` | The whole workflow, run with `invoke()`. Passing `checkpointer` makes the previously saved value (`previous`) available per `thread_id` |
| `entrypoint.final(value=..., save=...)` | Separates the value returned to the caller (value) from the value saved as `previous` for the next call (save) |

## Additional examples (corresponding to built-in features in 01_LangChain)

These examples use the same topics and flows as the `_PGVector` and `_Builtin` files in 01_LangChain.
They use `@task` and `@entrypoint` to implement the processing that 01_LangChain delegates to built-in middleware (`create_agent(middleware=[...])`).

| Base | Added file | Feature used (corresponding LangChain feature) |
| ------ | ---------------- | ------------------------------------ |
| `02_04_Retrieval.py` | `02_04_Retrieval_PGVector.py` | `PGVector` (langchain-postgres). Run `docker compose up -d` first |
| `02_05_MemoryThread.py` | `02_05_MemoryThread_Builtin.py` | Add a step summarizing older history when it grows long (`SummarizationMiddleware`) |
| `03_05_Recovery.py` | `03_05_Recovery_Builtin.py` | `@task(retry_policy=RetryPolicy(...))` (`ModelRetryMiddleware` / `ToolRetryMiddleware`) and `with_fallbacks()` (`ModelFallbackMiddleware`) |
| `04_03_PlanExecute.py` | `04_03_PlanExecute_Builtin.py` | Manually implement a ToDo-list tool and execution limit (`TodoListMiddleware` / `ToolCallLimitMiddleware`) |
| `04_06_GoalLoop.py` | `04_06_GoalLoop_Builtin.py` | Manually implement limits (`ToolCallLimitMiddleware` / `ModelCallLimitMiddleware`) |

There is no equivalent to `03_02_OutputCheck_Builtin.py` (`PIIMiddleware`), because its implementation here would be identical to `03_02_OutputCheck.py`.
