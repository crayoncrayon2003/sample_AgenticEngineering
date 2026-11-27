import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def main():
    """
    題材：架空の社内文書（data/docs）について質問に答える。
         質問は、合計4件ある。最後の1件は、文書に答えが書かれていない質問。
          1. 質問の文章を分解する。例えば、「今日は、天気がいい」-> ["今日","は","天気","が","いい"]
          2. 分解した文章をベクトル化する。例えば、["今日","は","天気","が","いい"]-> [0.3,0.4,0.2,0.1,0.8]
          3. 類似度を使って検索する。例えば、[0.3,0.4,0.2,0.1,0.8]に合う文章を探す。
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
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"回答：{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
