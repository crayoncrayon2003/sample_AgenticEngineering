import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic


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

llm = make_llm("minimal")

#####################################
# Langfuse：実行の記録（トレース）の送信先
#####################################
# ログの送信先を省略すると Langfuse Cloud（インターネット上）に送られる。ローカルの Langfuse に送信する
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

# 07_01_PromptManagement_01preprocess.py で登録したプロンプトの名前
PROMPT_NAME = "review_extract"


#####################################
# Langfuse からプロンプトを取り出して、チェーンを作る
#####################################
def make_chain(label: str):
    # get_prompt：ラベルを指定して、プロンプトを取り出す。
    #   cache_ttl_seconds：取り出したプロンプトを覚えておく秒数。0 にすると毎回取り出す（画面での変更がすぐ反映される）
    try:
        prompt = langfuse.get_prompt(PROMPT_NAME, label=label, type="chat", cache_ttl_seconds=0)
    except Exception as error:
        raise RuntimeError(
            f"プロンプト「{PROMPT_NAME}」（ラベル {label}）を取り出せません。"
            "先に 07_01_PromptManagement_01preprocess.py を実行してください。"
        ) from error

    # get_langchain_prompt：{{review}} を LangChain の {review} の形に直したメッセージの一覧を返す
    template = ChatPromptTemplate.from_messages(prompt.get_langchain_prompt())
    # metadata に langfuse_prompt を入れると、トレースのLLMの記録に、使ったプロンプトの版がひも付く
    # （画面の Prompts で、版ごとの利用回数・トークン数・時間を見られる）
    template.metadata = {"langfuse_prompt": prompt}
    return prompt, template | llm | StrOutputParser()


def main():
    """
    題材：商品レビューから良い点・気になる点を抜き出す（01_LangChain/01_01_Chain.py の1段目と同じ題材）。
         プロンプトをコードに書かず、Langfuse に登録したプロンプトを取り出して使う。
         事前に 07_01_PromptManagement_01preprocess.py を実行して、プロンプトを登録しておく。

    流れ：
        1. 07_01_PromptManagement_02process.py（手元のPC）が、production と staging のラベルで、
           それぞれ Langfuse（サーバ）からプロンプトを取り出す。
        2. 07_01_PromptManagement_02process.py（手元のPC）が、取り出したプロンプトで、同じレビューを整理して結果を比べる。

    試してみること：
        画面の「Prompts → review_extract」で、production のラベルを版2に付け替えてから、もう一度実行する。
        コードを変えていないのに、production の結果が、版2のプロンプトの結果に変わる。
    """

    # 商品レビュー
    review = (
        "このワイヤレスイヤホンは、音がクリアです。例えば、周りの雑音が、気になりません。"
        "充電ケースは、小さく持ち運びやすいのも気に入っています。"
        "ただ、片方だけで使用した場合、接続が切れることがあります。また、説明書も分かりにくかったです。"
    )
    print(f"レビュー：{review}")

    for label in ["production", "staging"]:
        prompt, chain = make_chain(label)
        print()
        print(f"--- ラベル {label}（版{prompt.version}）---")
        with langfuse.start_as_current_observation(name="レビューの整理", input={"review": review, "label": label}) as span:
            result = chain.invoke({"review": review}, config={"callbacks": [CallbackHandler()]})
            span.update(output=result)
            print(result)
            print(f"トレース：{langfuse.get_trace_url()}")

    # 送信は裏でまとめて行われる。プログラムを終える前に、残っている記録をすべて送る
    langfuse.flush()


if __name__ == "__main__":
    main()
