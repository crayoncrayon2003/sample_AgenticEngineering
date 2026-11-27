import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.func import entrypoint, task

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
DOCS_DIR = os.path.join(ROOT, "data", "docs")
COLLECTION_NAME = "sample_docs"  # ベクトルストアの中で、文書のまとまりを区別する名前

# 埋め込みベクトルは OpenAI にしかない機能のため、Claude では実行できない
if PROVIDER != "openai":
    sys.exit("このサンプルは埋め込みベクトル（OpenAI専用）を使うため、LLM_PROVIDER=openai で実行してください。")


def make_llm(effort: str):
    """LLM_PROVIDER に応じて、OpenAI または Claude のチャットモデルを作る。

    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL") or "gpt-5-nano",
            reasoning_effort=effort,
            use_responses_api=True,
            timeout=60,
            max_retries=1,
        )
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


llm = make_llm("low")

#####################################
# 文書を読み込み、ベクトルストア（PostgreSQL + pgvector）に保存する
#####################################
# 02_04_Retrieval.py との違い：
#   InMemoryVectorStore はプログラムの終了とともに消えるが、PGVector はデータベースに残る。
#   チャンクごとに決まったID（ファイル名-番号）を付けて保存するため、何度実行しても重複せず、上書きされる。
def build_vector_store() -> PGVector:
    # .env の SESSION_DATABASE_URL に保存する（事前に docker compose up -d を実行する）
    url = os.getenv("SESSION_DATABASE_URL")
    if not url:
        sys.exit(".env に SESSION_DATABASE_URL を設定してください。")
    # PGVector は SQLAlchemy 経由で接続するため、「postgresql+psycopg://」の形に直す
    connection = url.replace("postgresql://", "postgresql+psycopg://", 1)

    store = PGVector(
        # embeddings      : 文章をベクトルにする方法
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        # collection_name : 文書のまとまりの名前
        collection_name=COLLECTION_NAME,
        # connection      : 保存先のデータベース
        connection=connection,
        # use_jsonb       : メタデータ（出典など）をJSONB型で保存する
        use_jsonb=True,
    )

    # data/docs の .txt ファイルを、1件ずつ読み込む
    documents = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            documents.append(Document(page_content=file.read(), metadata={"source": filename}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_documents(documents)

    # チャンクごとのID：同じファイルの同じ位置のチャンクには、毎回同じIDを付ける
    ids = []
    counts = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counts[source] = counts.get(source, 0) + 1
        ids.append(f"{source}-{counts[source]}")

    store.add_documents(chunks, ids=ids)
    print(f"文書{len(documents)}件を、{len(chunks)}個のチャンクに分けて、PostgreSQL に保存しました。")
    return store


vector_store = build_vector_store()


#####################################
# 検索：社内文書から、質問に関係する部分を探す
#####################################
def search_documents(query: str) -> str:
    """社内文書から、質問に関係する部分を検索する。"""
    results = vector_store.similarity_search_with_score(query, k=3)
    print(f"  [検索] {query}")
    lines = []
    for doc, score in results:
        # PGVector のスコアは「距離」。InMemoryVectorStore の類似度とは逆に、小さいほど近い
        print(f"    - {doc.metadata['source']}（距離 {score:.2f}）")
        lines.append(f"【出典：{doc.metadata['source']}】\n{doc.page_content}")
    return "\n\n".join(lines) or "該当する文書はありません。"


#####################################
# 回答を作るチェーン
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは社内文書の案内係です。"
        "検索結果に書かれていることだけを使い、根拠にしたファイル名を出典として付けてください。"
        "検索結果で確認できない場合は『文書では確認できませんでした』と答え、推測で補わないでください。",
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "【質問】\n{question}\n\n【検索結果】\n{context}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# タスク（ワークフローの1ステップ）
#####################################
# 1段目：質問に関係する文書を検索する
@task
def retrieve(question: str) -> str:
    return search_documents(question)


# 2段目：検索結果だけを使って回答する
@task
def generate(question: str, context: str) -> str:
    return chain_answer.invoke({"question": question, "context": context})


#####################################
# ワークフロー：検索 → 回答
#####################################
@entrypoint()
def workflow(question: str) -> dict:
    result_step1 = retrieve(question).result()
    result_step2 = generate(question, result_step1).result()
    return {"context": result_step1, "answer": result_step2}


def main():
    """
    題材：架空の社内文書（data/docs）について質問に答える。
         質問は、合計4件ある（02_04_Retrieval.py と同じ）。最後の1件は、文書に答えが書かれていない質問。
         「検索 → 回答」の順番を、プログラムで決めている。
         02_04_Retrieval.py との違いは、ベクトルの保存先が PostgreSQL（pgvector）であること。
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
        # 1段目（検索）と2段目（回答）を、まとめて実行する
        result = workflow.invoke(question)
        print(f"回答：{result['answer']}")


if __name__ == "__main__":
    main()
