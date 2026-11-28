import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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



#####################################
# 1段目：レビューから要点を抜き出す
#####################################
prompt_extract = ChatPromptTemplate.from_template(
    "次の商品レビューから、良い点と気になる点を箇条書きで抜き出してください。\n"
    "レビューに書かれていないことは追加しないでください。\n\n"
    "{review}"
)
chain_extract = prompt_extract | llm | StrOutputParser()

#####################################
# 2段目：要点をJSONに整える
#####################################
prompt_json = ChatPromptTemplate.from_template(
    "次の要点を、キーが good（配列）と concerns（配列）のJSONに変換してください。\n"
    "JSONだけを出力してください。\n\n"
    "{points}"
)
chain_json = prompt_json | llm | JsonOutputParser()

#####################################
# 2つをつないだ1本のチェーン ---
#####################################
# 1段目の出力を、2段目のプロンプトの {points} に入れる
chain_all = {"points": chain_extract} | chain_json


def main():
    """題材：商品レビューから良い点・気になる点を抜き出し、JSONに整える。"""

    # 商品レビュー
    review = (
        "このワイヤレスイヤホンは、音がクリアです。例えば、周りの雑音が、気になりません。"
        "充電ケースは、小さく持ち運びやすいのも気に入っています。"
        "ただ、片方だけで使用した場合、接続が切れることがあります。また、説明書も分かりにくかったです。"
    )
    print(f"レビュー：{review}")
    
    # ##################################################################
    # 方法1：順番に呼び出し、途中の結果を確認する
    # ##################################################################
    # Langfuse：invoke を2回呼ぶと2つのトレースになるため、with の中の実行を1つのトレースにまとめる
    with langfuse.start_as_current_observation(name="方法1：順番に呼び出す"):
        # Step1 良い点・気になる点を抜き出す
        result_step1 = chain_extract.invoke({"review": review}, config={"callbacks": [CallbackHandler()]})
        print()
        print("--- 1段目：抜き出した要点 ---")
        print(result_step1)

        # Step2 jsonに変換する
        result_step2 = chain_json.invoke({"points": result_step1}, config={"callbacks": [CallbackHandler()]})
        print()
        print("--- 2段目：JSON ---")
        print(result_step2)

    # ##################################################################
    # 方法2：1本のチェーンとして、 良い点・気になる点を抜き出しJSONに変換する
    # ##################################################################
    print()
    print("--- 1本のチェーンで実行 ---")
    print(chain_all.invoke({"review": review}, config={"callbacks": [CallbackHandler()]}))

    # Langfuse：記録は裏でまとめて送られるため、プログラムを終える前に、残りの記録をすべて送る
    langfuse.flush()


if __name__ == "__main__":
    main()
