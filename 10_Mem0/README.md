# 10_Mem0

会話から利用者についての事実を抜き出し、長期記憶として保存・検索するライブラリ mem0 のサンプルです。

01_LangChain と同じ例の、長期記憶の部分を mem0 に置き換えています。

01_LangChain と 10_Mem0 の同じ番号のファイルを比較すると、違いが見えます。

## 準備

```bash
cp 10_Mem0/.env.example 10_Mem0/.env
```

| 変数 | 必須 | 内容 |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |

mem0 の利用状況の送信（テレメトリ）は、`MEM0_TELEMETRY=False` を設定して止めています。

## 事実の抜き出しに使うLLM

mem0 は、何も指定しないと OpenAI を直接呼んで、発言から事実を抜き出します。

ただし、02_07_MemoryUserProfile.py は、`make_llm` で作った LangChain のモデルを、mem0 の設定（`llm` の `provider: langchain`）で渡しています。

このため LLM_PROVIDER で OpenAI と Claude を切り替えられます。

なお、記憶を検索するための埋め込みベクトルは、OpenAI（`text-embedding-3-small`）を使います。

## 注意

- 埋め込みベクトルは OpenAI にしかない機能のため、`LLM_PROVIDER=openai` で実行します（01_LangChain/02_07 と同じ）。
- 記憶は、Qdrant に、プログラムの実行中だけメモリに置きます（01_LangChain/02_07 の `InMemoryStore` と同じ）。
- mem0 の設定（`config.json`）と記憶の変更履歴（`history.db`）は、既定のホームフォルダ（`~/.mem0`）ではなく、`10_Mem0/outputs/mem0/` に置きます。
- 事実の抜き出し方は、`custom_instructions` で指示しています（日本語の「利用者は〜」の形にする、質問だけの発言からは抜き出さない）。
