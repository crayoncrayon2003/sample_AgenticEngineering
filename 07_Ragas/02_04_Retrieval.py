import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Ragas の利用状況の送信（テレメトリ）を止める。Ragas を import する前に設定する
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance, Faithfulness

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic

# 埋め込みベクトルは OpenAI にしかない機能のため、Claude では実行できない
if PROVIDER != "openai":
    sys.exit("このサンプルは埋め込みベクトル（OpenAI専用）を使うため、LLM_PROVIDER=openai で実行してください。")

DOCS_DIR = os.path.join(ROOT, "data", "docs")


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


#####################################
# Ragas の評価に使うLLM
#####################################
# Ragas の評価指標は、LangChain のモデルではなく、Ragas 用のLLM（llm_factory で作る）を使う。LLM_PROVIDER に応じて作り分ける
#   max_tokens：推論するモデルは、考える分のトークンも使うため、多めにしておく（足りないと評価が途中で止まる）
def make_ragas_llm():
    if PROVIDER == "openai":
        from openai import OpenAI

        return llm_factory(os.getenv("OPENAI_MODEL") or "gpt-5-nano", client=OpenAI(), max_tokens=16000)
    from anthropic import Anthropic

    return llm_factory(os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5", provider="anthropic", client=Anthropic(), max_tokens=16000)


llm = make_llm("low")


#####################################
# 文書を読み込み、ベクトルストアを作る
#####################################
def build_vector_store() -> InMemoryVectorStore:
    # data/docs の .txt ファイルを、1件ずつ読み込む
    documents = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), encoding="utf-8-sig") as file:
            documents.append(Document(page_content=file.read(), metadata={"source": filename}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    print(f"文書{len(documents)}件を、{len(chunks)}個のチャンクに分けました。")

    store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents(chunks)
    return store


vector_store = build_vector_store()

# 評価のために、検索で見つかった文書を記録しておく（質問ごとに空にする）
retrieved_contexts = []


#####################################
# 検索ツール
#####################################
@tool
def search_documents(query: str) -> str:
    """社内文書から、質問に関係する部分を検索する。

    Args:
        query: 調べたい内容を表す検索文
    """
    results = vector_store.similarity_search_with_score(query, k=3)
    print(f"  [検索] {query}")
    lines = []
    for doc, score in results:
        print(f"    - {doc.metadata['source']}（類似度 {score:.2f}）")
        lines.append(f"【出典：{doc.metadata['source']}】\n{doc.page_content}")
        retrieved_contexts.append(doc.page_content)
    return "\n\n".join(lines) or "該当する文書はありません。"


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[search_documents],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは社内文書の案内係です。答える前に必ずsearch_documentsで調べてください。"
        "検索結果に書かれていることだけを使い、根拠にしたファイル名を出典として付けてください。"
        "検索結果で確認できない場合は『文書では確認できませんでした』と答え、推測で補わないでください。"
    ),
)


#####################################
# 評価：Ragas の評価指標で、回答と検索結果を評価する
#####################################
# Faithfulness     ：回答の内容が、検索結果に書かれていることだけでできているか（0〜1。1 は、すべて検索結果で裏付けられる）
# ContextRelevance ：検索結果が、質問に関係しているか（0〜1）
ragas_llm = make_ragas_llm()
metric_faithfulness = Faithfulness(llm=ragas_llm)
metric_relevance = ContextRelevance(llm=ragas_llm)


def main():
    """
    題材：
        社内文書を検索して答えるエージェント（RAG）が、正しく動くかを、Ragas の評価指標で確かめる。
        回答は文章で、正解が1つに決まらないため、LLMに採点させる。

    評価の対象：
        エージェントは、質問を受けると、search_documents で社内文書（data/docs）を検索し、検索結果だけを使って答える。
        検索は、文書と質問を埋め込みベクトルにして、意味が近い文書を探す。

    質問（4件）：
        1. 第2四半期の売上目標と重点地域
        2. 研究室Xで薬品を扱うときの人数と、利用できる時間帯
        3. 営業施策と安全管理の担当者
        4. 研究室Xの電話番号（文書に答えが書かれていない質問）

    判定すること：
        質問ごとに、回答と検索結果を、採点を担当するLLMに、2つの基準で採点させる。
          1. Faithfulness    ：回答の内容が、検索結果に書かれていることだけでできているか（推測で補っていないか）
          2. ContextRelevance：検索結果が、質問に関係しているか（検索がうまくいったか）
        点数は、0〜1 の範囲である。0 は「まったく満たしていない」、1 は「完全に満たしている」に該当する。
        採点を担当するLLMは、make_ragas_llm で作る（Ragas は、LangChain のモデルではなく、Ragas 用のLLMを使う）。

    比べ方：
        合格・不合格は判定しない。質問ごとに、2つの点数を表示する。

    流れ（質問1件分）：
        1. エージェントが、社内文書を検索して答える。検索で見つかった文書は、retrieved_contexts に記録しておく。
        2. Ragas の評価指標が、質問・回答・検索結果を受け取り、2つの基準で採点する。
        3. 結果を表示する。

    01_LangChain/02_04_Retrieval.py との違い：
        評価（判定すること）を加えた点だけが違う。01_LangChain は回答を表示するだけで、評価はしない。
        エージェント・検索・質問は同じ。

    注意：
        埋め込みベクトルは OpenAI にしかない機能のため、LLM_PROVIDER=openai で実行する。
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
        # エージェントを実行する（"user"：アプリを使う利用者の発言）
        retrieved_contexts.clear()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"回答：{result['messages'][-1].text}")

        # 回答と、検索で見つかった文書を評価する
        faithfulness = metric_faithfulness.score(user_input=question, response=result["messages"][-1].text, retrieved_contexts=retrieved_contexts)
        relevance = metric_relevance.score(user_input=question, retrieved_contexts=retrieved_contexts)
        print(f"評価：回答が検索結果に基づく度合い {faithfulness.value:.2f}、検索結果と質問の関係の強さ {relevance.value:.2f}")


if __name__ == "__main__":
    main()
