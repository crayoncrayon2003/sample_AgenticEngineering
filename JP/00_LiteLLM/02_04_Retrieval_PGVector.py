import json
import os
import sys

import litellm
import psycopg
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
DOCS_DIR = os.path.join(ROOT, "data", "docs")
TABLE_NAME = "doc_chunks"  # チャンクとベクトルを保存するテーブル

# 埋め込みベクトルは OpenAI にしかない機能のため、Claude では実行できない
if PROVIDER != "openai":
    sys.exit("このサンプルは埋め込みベクトル（OpenAI専用）を使うため、LLM_PROVIDER=openai で実行してください。")


def make_llm(effort: str) -> dict:
    """LLM_PROVIDER に応じて、LiteLLM に渡すモデルの設定を作る。

    LiteLLM は「提供元/モデル名」の文字列で、呼び出すLLMを切り替える。
    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """LiteLLM でLLMを呼び、返ってきたメッセージを返す（LangChain の llm.invoke に当たる）。"""
    response = litellm.completion(
        model=llm["model"],
        reasoning_effort=llm["reasoning_effort"],
        max_tokens=llm["max_tokens"],
        messages=messages,
        response_format=response_format,
        tools=tools,
        timeout=60,
        num_retries=1,
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """プロンプトの {名前} に値を入れる（LangChain の ChatPromptTemplate に当たる）。"""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")


#####################################
# 文書を読み込み、ベクトル化する
#####################################
# 文章を、120文字ほどの塊（チャンク）に分ける（LangChain の RecursiveCharacterTextSplitter に当たる簡易版）
def split_text(text: str, chunk_size: int = 120) -> list[str]:
    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) > chunk_size:
            chunks.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# 文章を埋め込みベクトル（数値の並び）に変える
def embed(texts: list[str]) -> list[list[float]]:
    response = litellm.embedding(model="openai/text-embedding-3-small", input=texts)
    return [item["embedding"] for item in response.data]


# ベクトルを、pgvector が受け取れる文字列（例：[0.1,0.2,0.3]）に直す
def to_vector_text(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"


# .env の SESSION_DATABASE_URL に接続する（事前に docker compose up -d を実行する）
def connect() -> psycopg.Connection:
    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit(".env に SESSION_DATABASE_URL を設定してください。")
    return psycopg.connect(url, autocommit=True)


# 02_04_Retrieval.py との違い：
#   02_04_Retrieval.py はベクトルをPythonのリストに持つため、プログラムの終了とともに消える。
#   ここでは PostgreSQL のテーブルに保存する（LangChain の PGVector に当たる）。
#   チャンクごとに決まったID（ファイル名-番号）を付け、同じIDがあれば上書きするため、何度実行しても重複しない。
def build_vector_store() -> None:
    # data/docs の .txt ファイルを、1件ずつ読み込んで、チャンクに分ける
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            for number, text in enumerate(split_text(file.read()), start=1):
                chunks.append({"id": f"{filename}-{number}", "source": filename, "text": text})

    vectors = embed([chunk["text"] for chunk in chunks])
    with connect() as conn:
        # pgvector の拡張を有効にし、vector 型（1536次元：text-embedding-3-small の大きさ）の列を持つテーブルを作る
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} "
            "(id text PRIMARY KEY, source text NOT NULL, text text NOT NULL, embedding vector(1536) NOT NULL)"
        )
        # 同じIDがあれば上書きする（ON CONFLICT ... DO UPDATE）
        for chunk, vector in zip(chunks, vectors):
            conn.execute(
                f"INSERT INTO {TABLE_NAME} (id, source, text, embedding) VALUES (%s, %s, %s, %s::vector) "
                "ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source, text = EXCLUDED.text, embedding = EXCLUDED.embedding",
                (chunk["id"], chunk["source"], chunk["text"], to_vector_text(vector)),
            )
    print(f"{len(chunks)}個のチャンクに分けて、PostgreSQL に保存しました。")


build_vector_store()


#####################################
# 検索ツール
#####################################
def search_documents(query: str) -> str:
    """社内文書から、質問に関係する部分を検索する。"""
    query_vector = embed([query])[0]
    # <=> ：pgvector のコサイン距離。小さいほど近い（02_04_Retrieval.py の類似度とは大小が逆）
    with connect() as conn:
        rows = conn.execute(
            f"SELECT source, text, embedding <=> %s::vector AS distance FROM {TABLE_NAME} ORDER BY distance LIMIT 3",
            (to_vector_text(query_vector),),
        ).fetchall()
    print(f"  [検索] {query}")
    lines = []
    for source, text, distance in rows:
        print(f"    - {source}（距離 {distance:.2f}）")
        lines.append(f"【出典：{source}】\n{text}")
    return "\n\n".join(lines) or "該当する文書はありません。"


# LLMに渡すツールの説明
tools = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "社内文書から、質問に関係する部分を検索する。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "調べたい内容を表す検索文"}},
            "required": ["query"],
        },
    }},
]


#####################################
# エージェント
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = (
    "あなたは社内文書の案内係です。答える前に必ずsearch_documentsで調べてください。"
    "検索結果に書かれていることだけを使い、根拠にしたファイル名を出典として付けてください。"
    "検索結果で確認できない場合は『文書では確認できませんでした』と答え、推測で補わないでください。"
)


def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    message = call_llm(llm, messages, tools=tools)

    # LLMがツール呼び出しを返す間は、ツールを実行して、もう一度LLMを呼ぶ
    while message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = search_documents(args["query"])
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        message = call_llm(llm, messages, tools=tools)

    return message.content


def main():
    """
    題材：架空の社内文書（data/docs）について質問に答える。
         質問は、合計4件ある。最後の1件は、文書に答えが書かれていない質問。
          1. 文書を分解する（チャンクに分ける）。
          2. 分解した文章をベクトル化する。
          3. 質問もベクトル化し、PostgreSQL（pgvector）で距離が近い文章を探す。
         02_04_Retrieval.py との違いは、ベクトルの保存先と検索を PostgreSQL に任せていること。
    """

    # ４件の質問
    questions = [
        "第2四半期の売上目標と重点地域を教えてください。",
        "研究室Xで薬品を扱うときの人数と、利用できる時間帯は？",
        "営業施策と安全管理の担当者をそれぞれ教えてください。",
        "研究室Xの電話番号を教えてください。",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f"質問：{question}")
        # エージェントを実行する
        answer = run_agent(question)
        print(f"回答：{answer}")


if __name__ == "__main__":
    main()
