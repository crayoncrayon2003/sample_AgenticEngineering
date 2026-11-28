# 07_Ragas

RAG や LLMアプリの回答を、テストケースと評価指標（メトリクス）で書けるようにするライブラリ Ragas のサンプルです。

01_LangChain と同じ例に、Ragas による評価を加えています。

01_LangChain と 07_Ragas の同じ番号のファイルを比較すると、違いが見えます。

## 準備

```bash
cp 07_Ragas/.env.example 07_Ragas/.env
```

| 変数 | 必須 | 内容 |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |

Ragas の利用状況の送信（テレメトリ）は、各ファイルで `RAGAS_DO_NOT_TRACK=true` を設定して止めています。

## 採点に使うLLM

Ragas の評価指標は、採点に使うLLMを必ず指定します。

LangChain のモデルは渡せず、Ragas 用のLLM（`llm_factory` で作る）を渡します。

02_04_Retrieval.py と 03_09_EvalJudge.py は、`make_ragas_llm` で、`LLM_PROVIDER` に応じた OpenAI か Claude の Ragas 用のLLMを作って渡しています。

このため LLM_PROVIDER で OpenAI と Claude を切り替えられます。

## 注意

- `02_04_Retrieval.py` は、01_LangChain/02_04 と同じく埋め込みベクトル（OpenAI専用）を使うため、`LLM_PROVIDER=openai` で実行します。
- 推論するモデルは考える分のトークンも使うため、`max_tokens` を多めにしています。少ないと、評価が途中で止まります。

## DeepEval と Ragas の違い

LLMを含むソフトウェアのテストの観点で、次の２つはほぼ同じ。出発点・目的が異なる。

* DeepEval : 発想の出発点はソフトウェアテスト。
* Ragas    : 発想の出発点はRAGの評価。

