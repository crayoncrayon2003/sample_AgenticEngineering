# 02_LangGraph

01_LangChain と同じ例を、LangGraph で書き直したサンプルです。

プロンプトとチェーンは 01_LangChain と同じです。

異なる点は「処理の流れの書き方」だけです。

01_LangChain と 02_LangGraph の 同じ番号のファイルを比較すると違いが見えます。

## Functional API の考え方

| 部品 | 役割 |
|---|---|
| `@task` | ワークフローの1ステップ。呼ぶとすぐに「結果の予約券」が返る。`.result()` で結果を受け取る。`.result()` を呼ぶ前に別のタスクを呼べば、同時に実行される |
| `@entrypoint` | ワークフロー全体。`invoke()` で実行する。`checkpointer` を渡すと、`thread_id` ごとに前回の保存値（`previous`）を受け取れる |
| `entrypoint.final(value=..., save=...)` | 呼び出し元に返す値（value）と、次回の `previous` として保存する値（save）を分ける |

## 追加のサンプル（01_LangChain の標準機能版に対応）

01_LangChain の `_PGVector`・`_Builtin` のファイルと、同じ題材・同じ流れのサンプルです。
01_LangChain が標準ミドルウェア（`create_agent(middleware=[...])`）に任せている処理を、`@task`・`@entrypoint`で組み立てています。

| ベース | 追加したファイル | 使う機能（LangChain の対応する機能） |
| ------ | ---------------- | ------------------------------------ |
| `02_04_Retrieval.py` | `02_04_Retrieval_PGVector.py` | `PGVector`（langchain-postgres）。事前に `docker compose up -d` が必要 |
| `02_05_MemoryThread.py` | `02_05_MemoryThread_Builtin.py` | 履歴が長くなったら、古い部分を要約するステップを加える（`SummarizationMiddleware`） |
| `03_05_Recovery.py` | `03_05_Recovery_Builtin.py` | `@task(retry_policy=RetryPolicy(...))`（`ModelRetryMiddleware`・`ToolRetryMiddleware`）と `with_fallbacks()`（`ModelFallbackMiddleware`） |
| `04_03_PlanExecute.py` | `04_03_PlanExecute_Builtin.py` | ToDoリストのツールと、実行回数の上限を自分で書く（`TodoListMiddleware`・`ToolCallLimitMiddleware`） |
| `04_06_GoalLoop.py` | `04_06_GoalLoop_Builtin.py` | 自分で書く（`ToolCallLimitMiddleware`・`ModelCallLimitMiddleware`） |

`03_02_OutputCheck_Builtin.py`（`PIIMiddleware`）に当たるファイルはありません。このフォルダで書くと、`03_02_OutputCheck.py` と同じ内容になるためです。
