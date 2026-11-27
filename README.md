# 準備

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## .env の作成

各フォルダで `.env.example` を `.env` にコピーする

`.env`に、APIキーなどを設定する。

```bash
cp 00_LiteLLM/.env.example 00_LiteLLM/.env
```

## データベースの起動

次のサンプルを動かす場合は、PostgreSQL を起動する。

- `02_05_MemoryThread.py`：保存先に postgres を指定する場合
- `02_04_Retrieval_PGVector.py`（各フォルダ）：ベクトルの保存先に使う

```bash
docker compose up -d
```

イメージは、ベクトル検索の拡張機能を追加した、PostgreSQL 17 （`pgvector/pgvector:pg17`） を使う。

`.env` の `SESSION_DATABASE_URL` には次を設定する。

```
SESSION_DATABASE_URL=postgresql://agent_user:local_dev_password@127.0.0.1:15432/agent_sessions?sslmode=disable
```

## Langfuse の起動

`05_Langfuse` のサンプルを動かす場合は、Langfuse をローカルで起動する。キーの取得は不要。詳しくは [05_Langfuse/README.md](05_Langfuse/README.md) を参照。

```bash
docker compose -f 05_Langfuse/docker-compose.yaml up -d
```

# APIキー発行

このサンプルを動かすためには、LLMを呼び出すためのAPIキーが必要になる。

## OpenAI

[https://platform.openai.com](https://platform.openai.com)

### チャージ

1. 左側メニュー **HOME** を選択する
2. **Add Credits**からチャージする。

### APIキー発行

1. 左側メニューから **API keys** を選択する
2. **Create new secret key**を選択してAPIキーを発行する。

## Anthropic

[https://platform.claude.com](https://platform.claude.com)

### チャージ

1. 設定から **Billing** を選択する。
2. **Buy credits** を選択し、購入金額を入力してチャージする。

### APIキー発行

1. **API keys** を選択する。
2. **Create key** を選択し、名前などを設定してAPIキーを発行する。
3. 表示されたAPIキーをコピーして保存する。
