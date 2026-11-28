# 06_DeepEval

LLMアプリの評価を、テストケースと評価指標（メトリクス）で書けるようにするライブラリ DeepEval のサンプルです。

01_LangChain と同じ例の、評価の部分を DeepEval に置き換えています。

01_LangChain と 06_DeepEval の同じ番号のファイルを比較すると、違いが見えます。

## 準備

```bash
cp 06_DeepEval/.env.example 06_DeepEval/.env
```

| 変数 | 必須 | 内容 |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |

DeepEval の利用状況の送信（テレメトリ）は、各ファイルで `DEEPEVAL_TELEMETRY_OPT_OUT=YES` を設定して止めています。

## 採点に使うLLM

DeepEval は、何も指定しないと OpenAI を直接呼んで採点します。

ただし、03_09_EvalJudge.pyは、`make_llm` で作った LangChain のモデルを `DeepEvalBaseLLM` で包んで渡しています

このため LLM_PROVIDER で OpenAI と Claude を切り替えられます。

## 注意

- `GEval` の点数は 0〜1点です（01_LangChain/03_09 は1〜5点）。
- `GEval` の返却する採点理由は、英語になることがあります。DeepEval の採点用のプロンプトが英語のため。

## DeepEval と Ragas の違い

LLMを含むソフトウェアのテストの観点で、次の２つはほぼ同じ。出発点・目的が異なる。

* DeepEval : 発想の出発点はソフトウェアテスト。
* Ragas    : 発想の出発点はRAGの評価。

