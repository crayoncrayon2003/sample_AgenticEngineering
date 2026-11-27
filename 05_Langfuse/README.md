# 05_Langfuse

01_LangChain と同じ例に、Langfuse によるトレース（実行の記録）・評価・プロンプト管理を加えたサンプルです。

Langfuse はオープンソースのため、自分のPCの Docker で動かせます。キーの取得やアカウントの登録は不要です。

01_LangChain と 05_Langfuse の同じ番号のファイルを比較すると、Langfuse を使うために加えた部分が見えます。

## 準備

### 1. Langfuse を起動する

次を実行します。初回は、イメージのダウンロードに数分かかります。

```bash
docker compose docker-compose.yaml up
```

6つのコンテナ（GUI・処理・PostgreSQL・ClickHouse・Redis・MinIO）が起動します。

メモリを数GB使います。

起動したら、ブラウザで [http://localhost:3000](http://localhost:3000) を開きます。

次の情報でログインします。

| 項目 | 値 |
| ---- | -- |
| メールアドレス | `admin@example.com` |
| パスワード | `local_dev_password` |

左側メニュー 「Projects」から、「Agentic Engineering Sample」の「Go to project」を選択する。

左側メニュー 「Tracing」から、該当のログを見つけ出して、クリックすると内容が確認できる。

使い終わったら、次で停止します。記録したデータは Docker のボリュームに残り、次回の起動時にも使えます。

```bash
docker compose -f 05_Langfuse/docker-compose.yaml down
```

データも消す場合は、`down -v` を使います。

### 2. .env を作る

```bash
cp 05_Langfuse/.env.example 05_Langfuse/.env
```

| 変数 | 必須 | 内容 |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | 使うLLMの提供元。`openai` または `anthropic`。省略時は `openai` |
| `OPENAI_API_KEY` | OpenAIを使う場合 | OpenAIのAPIキー |
| `OPENAI_MODEL` | | OpenAIのモデル。省略時は `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | Claudeを使う場合 | AnthropicのAPIキー |
| `ANTHROPIC_MODEL` | | Claudeのモデル。省略時は `claude-haiku-5-5` |
| `LANGFUSE_BASE_URL` | | Langfuse の接続先。省略時は `http://localhost:3000` |
| `LANGFUSE_PUBLIC_KEY` | ○ | Langfuse の公開キー。`.env.example` の値（`docker-compose.yaml` で作ったキー）のままでよい |
| `LANGFUSE_SECRET_KEY` | ○ | Langfuse の秘密キー。`.env.example` の値のままでよい |

`LANGFUSE_BASE_URL` を省略すると、Langfuse の SDK はインターネット上の Langfuse Cloud に送ろうとします。
このフォルダのサンプルは、省略時の送り先を `http://localhost:3000` にしているため、ローカルの Langfuse にだけ送ります。

## トレースの付け方

### 必須の形

次の2つで、LangChain の実行を記録します。

| 部品 | 役割 |
| ---- | ---- |
| `CallbackHandler()` | LangChain の実行を記録し、Langfuse に送る |
| `langfuse.flush()` | プログラムが終了すると記録は失われる。最後に残りをすべて送る。 |

```python
from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

result = agent.invoke(
    { "messages": [...] },
    config = {"callbacks": [CallbackHandler()]}
)

langfuse.flush()
```


### 任意の形

必須の形だけでも記録されます。

`invoke` 1回ごとに別々のトレースになり、名前も自動で付きます。

次の部品を使うと、トレースをまとめたり、名前やタグを付けられるため、ログを探しやすくしたりできます。

| 部品 | 役割 |
| ---- | ---- |
| `langfuse.start_as_current_observation(name=...)` | 複数の実行を、1つのトレースにまとめる。トレースの入力・出力も記録できる | 
| `config` の `metadata` に `langfuse_session_id` | 同じ会話のトレースを、1つのセッションにまとめる（02_05） |
| `config` の `metadata` に `langfuse_tags` | トレースにタグを付ける（03_07） |
| `propagate_attributes(...)` | トレースに、名前・タグ・セッションID・メタデータを付ける |
| `langfuse.get_trace_url()` | 今のトレースをGUIで開くURLを返す |

### 本サンプル対象外の機能
#### 費用の表示

langfuseは、LLM の呼び出しにかかった費用を、トレースごと・モデルごとに確認できる機能を持つ。

ただし、本サンプルは、GUI に、利用するモデル・料金を登録する手順を省いている。
GUI の Settings -> Models で、利用するモデル・料金を登録すれば、分析時に費用が表示される。

#### Langfuse 側で実行

Langfuseは、 Langfuseのみで、LLM を呼び出し プロンプトを試行・検証できる。

本サンプルは、LLM の呼び出しも採点も、すべてローカル PC で行い、Langfuse は結果を保存・表示する。

Settings で、LLM の 情報を設定（URL、APIキー など）すれば、Langfuse 側で 次の機能が利用できる。

**LLM による自動評価（LLM-as-a-Judge）**

 ある質問に対する LLM の回答に対し、別のLLMを使って、そのの良し悪しをに判定させて、点数をスコアとして記録する。
 
 プログラムを書かずに Langfuse の GUI の設定だけで判定できる。

**プレイグラウンド**

GUI でプロンプトを書き換えながら、LLM の回答を試す。


#### ダッシュボード
アプリの利用状況や品質の変化を、グラフに可視化できる。

トレースの件数・時間・トークン数・費用・スコアの推移を、期間を指定して確認できる。

このほか、注釈キュー（Annotation Queues）を使うと、人が回答を確認して採点し、その結果をスコアとして残すこともできる。

どちらも GUI で直接操作して使う機能で、本サンプルにはコードがありません。
