# 09_GuardrailsAI

LLMの出力を、検査の部品（Validator）や Pydantic のクラスで検証するライブラリ Guardrails AI のサンプルです。

01_LangChain と同じ例の、検査の部分を Guardrails AI に置き換えています。

01_LangChain と 09_GuardrailsAI の同じ番号のファイルを比較すると、違いが見えます。

## 準備

このフォルダのサンプルは、ルートの仮想環境（`.venv`）ではなく、専用の仮想環境で動かします。

Guardrails AI は openai のパッケージ 2.x を必要とし、ルートの仮想環境（openai 3.x）と同時に入れられないためです。

専用の仮想環境には、このフォルダの `requirements.txt` のパッケージを入れます（ルートの `requirements.txt` には、Guardrails AI を含めていません）。

```bash
cp 09_GuardrailsAI/.env.example 09_GuardrailsAI/.env
```

| 変数 | 必須 | 内容 |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |

Guardrails AI の利用状況の送信（テレメトリ）は、各ファイルで止めています（次の「利用状況の送信について」を参照）。

## 検査に使うLLM

Guardrails AI は、LLMの呼び出しを Guard に任せて、出力を検査・修正させることもできます。

ただし、03_02_OutputCheck.py と 03_03_SchemaCheck.py は、LangChain のチェーンが作った出力を、Guard で検査するだけです。検査は自作の Validator と Pydantic のクラスで判定するため、LLMは使いません。

Guardrails Hub（既製の Validator の配布サービス）は使いません。Validator はすべて自作しているため、Hub への登録やトークンは不要です。

## 利用状況の送信について

Guardrails AI は、設定ファイル（`~/.guardrailsrc`）が無いと、利用状況をインターネット上のサーバーに送ります。

各ファイルでは、次の2か所で送信を止めています。

- `settings.rc.enable_metrics = False`：Validator を作る前に設定する（Validator は、作るときの設定を使う）
- `guard.configure(allow_metrics_collection=False)`：Guard ごとに設定する（Guard は、作るときに設定ファイルを読み直す）

`guardrails configure` コマンドで、設定ファイルに送信しない設定を書くこともできます。

## 注意

- `03_03_SchemaCheck.py` の「1995-02-30」のように、形（YYYY-MM-DD）は合っていても存在しない日付は、Pydantic のクラスの条件だけでは見つけられません。そのため、自作の Validator で検査しています。

## NeMo Guardrails と Guardrails AI の違い

LLMの入力・出力を検査する観点で、次の２つはほぼ同じ。出発点・目的が異なる。

* NeMo Guardrails : 発想の出発点は会話の制御。入力・出力・対話の流れに、設定（YAML）で組み立てたレールをかける。
* Guardrails AI   : 発想の出発点は出力の検証。Validator を組み合わせて、出力の形と中身を検証する。
