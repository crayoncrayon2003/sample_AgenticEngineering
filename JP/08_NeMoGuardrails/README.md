# 08_NeMoGuardrails

LLMへの入力や、LLMの出力を、設定（YAML）で組み立てた「レール」で検査するライブラリ NeMo Guardrails（NVIDIA）のサンプルです。

01_LangChain と同じ例の、検査の部分を NeMo Guardrails に置き換えています。

01_LangChain と 08_NeMoGuardrails の同じ番号のファイルを比較すると、違いが見えます。

## 準備

```bash
cp 08_NeMoGuardrails/.env.example 08_NeMoGuardrails/.env
```

| 変数 | 必須 | 内容 |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |

NeMo Guardrails の利用状況の送信（テレメトリ）は、各ファイルで `NEMO_GUARDRAILS_NO_USAGE_STATS=1` を設定して止めています。

## 審査に使うLLM

NeMo Guardrails は、通常、設定（YAML）の `models` に、審査に使うLLMを書きます。

ただし、03_01_InputCheck.py と 03_02_OutputCheck.py は、`make_llm` で作った LangChain のモデルを `LLMRails` に渡しています。

このため LLM_PROVIDER で OpenAI と Claude を切り替えられます。

なお、03_02_OutputCheck.py の検査（`regex check output`）は正規表現で判定するため、LLMは使いません。

## 03_04（ツール引数の検査）が無い理由

NeMo Guardrails にも、ツールの呼び出しを検査するレール（`tool_call`）はありますが、次の理由で、01_LangChain/03_04 と同じ例は書けません。

- 新しい実行エンジン（IORails）でしか動かず、このフォルダで使っている `LLMRails` では動かない
- 検査の式に使う CEL（式言語）の追加パッケージが必要
- 03_04 の要点である「ログイン中の利用者のIDと比べる」ために必要な、実行時の情報を検査に使えない

## 注意

- `self check input` は、許可・拒否（Yes・No）だけを返します。違反したルールや理由は返しません。
- `regex check output` は、どのパターンに一致したかまでは返しません。
- レールの設定は、各ファイルの `RAILS_CONFIG`（YAML）に書いています。ファイル（`config.yml`）に書いて `RailsConfig.from_path` で読み込むこともできます。

## NeMo Guardrails と Guardrails AI の違い

LLMの入力・出力を検査する観点で、次の２つはほぼ同じ。出発点・目的が異なる。

* NeMo Guardrails : 発想の出発点は会話の制御。入力・出力・対話の流れに、設定（YAML）で組み立てたレールをかける。
* Guardrails AI   : 発想の出発点は出力の検証。Validator を組み合わせて、出力の形と中身を検証する。
