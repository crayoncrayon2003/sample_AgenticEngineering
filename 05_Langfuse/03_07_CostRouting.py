import os
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
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


#####################################
# Langfuse：実行の記録（トレース）の送信先
#####################################
# ログの送信先を省略すると Langfuse Cloud（インターネット上）に送られる。ローカルの Langfuse に送信する
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

#####################################
# 2種類のモデル：軽いモデルと、高性能なモデル
#####################################
# 料金や速度は、提供元・時期によって変わります。最新の情報は、各社の料金ページで確認してください
if PROVIDER == "openai":
    MODEL_NAMES = {"light": "gpt-5-nano", "advanced": "gpt-5-mini"}
else:
    MODEL_NAMES = {"light": "claude-haiku-5-5", "advanced": "claude-sonnet-5-5"}

# 振り分けの基準：質問の文字数（文字数は難しさそのものではなく、教材用の簡易な基準）
CHARACTER_THRESHOLD = 100


# 指定した名前のモデルでLLMを作る（make_llm と同じ設定で、モデル名だけを変える）
def make_llm_with_model(model_name: str):
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, reasoning_effort="minimal", use_responses_api=True, timeout=60, max_retries=1)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model_name, reasoning_effort="low", max_tokens=16000, timeout=60, max_retries=1)


make_llm("minimal")  # APIキーの確認だけ行う

#####################################
# エージェント：モデルごとに1つずつ作る
#####################################
system_prompt = (
    "あなたはAIエージェントとデータ処理の技術解説係です。"
    "質問に日本語で分かりやすく答えてください。不明な仕様は推測で断定しないでください。"
)
agent_light = create_agent(model=make_llm_with_model(MODEL_NAMES["light"]), tools=[], system_prompt=system_prompt)
agent_advanced = create_agent(model=make_llm_with_model(MODEL_NAMES["advanced"]), tools=[], system_prompt=system_prompt)


# 質問の文字数で、使うエージェントを選ぶ
def select_route(question: str) -> str:
    if len(question) < CHARACTER_THRESHOLD:
        return "light"
    return "advanced"


def main():
    """
    題材：質問の長さで、軽いモデルと高性能なモデルを使い分け、費用と時間を抑える。
         質問は、合計2件ある。1件目は短い質問、2件目は長く複雑な質問。
    """

    print(f"振り分けの基準：{CHARACTER_THRESHOLD}文字未満なら light、それ以上なら advanced")

    # 2件の質問
    questions = [
        "ETLとは何ですか？",
        "複数のCSVファイルを毎日読み込み、列名の違い、文字コードの違い、欠損値、重複データを処理して、集計結果をJSONで保存したいです。途中で一部のファイルの読み込みに失敗した場合も考慮してください。処理の順序、各段階での検証、再試行と途中再開の方法を、設計上の注意点とともに説明してください。",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f"質問（{len(question)}文字）：{question}")

        # 1段目：使うモデルを選ぶ
        route = select_route(question)
        print(f"振り分け先：{route}（{MODEL_NAMES[route]}）")

        # 2段目：選んだモデルのエージェントで答え、時間と使用量を記録する
        if route == "light":
            agent = agent_light
        else:
            agent = agent_advanced
        start = time.perf_counter()
        # Langfuse：振り分け先をタグにして記録する（画面で、タグごとにトークン数・時間を比べられる）
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config={"callbacks": [CallbackHandler()], "metadata": {"langfuse_tags": [route]}})
        elapsed = time.perf_counter() - start

        answer = result["messages"][-1]
        usage = answer.usage_metadata or {}
        print(f"時間：{elapsed:.1f}秒、トークン数：入力 {usage.get('input_tokens')}、出力 {usage.get('output_tokens')}")
        print(f"回答：{answer.text}")

    # Langfuse：記録は裏でまとめて送られるため、プログラムを終える前に、残りの記録をすべて送る
    langfuse.flush()


if __name__ == "__main__":
    main()
