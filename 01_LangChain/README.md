# 01_LangChain

LangChainで書いたLLMを含むソフトウェアのサンプルです。

## 準備

`.env.example` をコピーして `01_LangChain/.env` を作り、値を設定します。

```bash
cp 01_LangChain/.env.example 01_LangChain/.env
```

| 変数                     |       必須       | 内容                                                                  |
| ------------------------ | :--------------: | --------------------------------------------------------------------- |
| `LLM_PROVIDER`           |                  | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY`         | OpenAIを使う場合 | OpenAIのAPIキー                                                       |
| `OPENAI_MODEL`           |                  | OpenAIのモデル。省略時は `gpt-5-nano`                               |
| `OPENAI_VECTOR_STORE_ID` |                  | 02_03 の file_search で使うベクトルストアのID                         |
| `ANTHROPIC_API_KEY`      | Claudeを使う場合 | AnthropicのAPIキー                                                    |
| `ANTHROPIC_MODEL`        |                  | Claudeのモデル。省略時は `claude-haiku-5-5`                         |
| `SESSION_DATABASE_URL`   |                  | 02_05 を PostgreSQL で動かす場合と、02_04_Retrieval_PGVector の接続先 |

値を空にした変数は、省略した場合と同じ扱いになります。

## OpenAI と Claude の切り替え

`.env` の `LLM_PROVIDER` で、OpenAI と Claudeを切り替えます。
サンプルによっては、OpenAI または Claude のいずれかのコードのみ存在するケースがあります。

## ミドルウェアの実行順序

`create_agent(middleware=[...])` に渡したミドルウェアは、`agent.invoke()` の中で次の順番に呼ばれます。
ツールを使う場合は、「LLM呼び出し → ツール実行」を、ツールが不要になるまで繰り返します。

```
agent.invoke() の開始
  ↓
@before_agent：開始時に1回
  ↓
@before_model
  ↓
LLM呼び出し（@dynamic_prompt・@wrap_model_call は、この呼び出しを包む）
  ↓
@after_model
  ↓
ツール呼び出しが必要？
  ├─ はい
  │    ↓
  │  @wrap_tool_call の実行前処理
  │    ↓
  │  handler(request)：ツール実行
  │    ↓
  │  @wrap_tool_call の実行後処理・例外処理
  │    ↓
  │  @before_model に戻る
  │
  └─ いいえ：最終回答
       ↓
@after_agent：正常終了時に1回
  ↓
agent.invoke() が結果を返す
```

| デコレーター         | 呼ばれるタイミング               | 主な使い道                                   |
| -------------------- | -------------------------------- | -------------------------------------------- |
| `@before_agent`    | `invoke()` の開始時に1回       | 入力のチェック、準備                         |
| `@before_model`    | LLMを呼ぶ前（毎回）              | 会話履歴の整理、指示の追加                   |
| `@dynamic_prompt`  | LLMを呼ぶとき（毎回）            | 状況に応じてシステムプロンプトを作る         |
| `@wrap_model_call` | LLM呼び出しを包む（毎回）        | 呼び出し前後の処理、再試行、モデルの切り替え |
| `@after_model`     | LLMの応答の後（毎回）            | 応答のチェック                               |
| `@wrap_tool_call`  | ツールの実行を包む（ツールごと） | 引数のチェック、例外の処理                   |
| `@after_agent`     | 最終回答の後に1回                | 結果の保存、後処理                           |
