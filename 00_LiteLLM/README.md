# 00_LiteLLM

LangChainで書いたLLMを含むソフトウェアのサンプルです。

## 準備

| 変数 | 必須 | 内容 |
|---|:-:|---|
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `OPENAI_VECTOR_STORE_ID` | | 02_03 のファイル検索で使うベクトルストアのID |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |
| `SESSION_DATABASE_URL` | | 02_05 を PostgreSQL で動かす場合と、02_04_Retrieval_PGVector の接続先 |
