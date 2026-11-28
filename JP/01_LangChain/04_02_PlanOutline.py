import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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


llm = make_llm("low")

#####################################
# LLMのレスポンス形式を定義する（構成案）
#####################################
class Outline(BaseModel):
    #  sections： 説明する項目を、説明する順に並べた一覧
    sections: list[str] = Field(description="説明する項目を、説明する順に3〜5個")


#####################################
# 1段目：文章の構成案を作るチェーン
#####################################
prompt_plan = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは技術解説文のライターです。文章を書く前に、説明する項目と順番を計画してください。"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "テーマ：{topic}"
    ),
])
chain_plan = prompt_plan | llm.with_structured_output(Outline, method="json_schema")

#####################################
# 2段目：構成案に沿って書くチェーン
#####################################
prompt_write = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは技術解説文のライターです。渡された構成案の順番に沿って、600字程度の解説文を日本語で書いてください。"
        "項目ごとに見出しを付けてください。"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "テーマ：{topic}\n\n【構成案】\n{outline}"
    ),
])
chain_write = prompt_write | llm | StrOutputParser()


def main():
    """
    題材：文章の構成案を作る。その構成案に沿って解説文を書く。
    """

    topic = "AIにおける強化学習の重要性"
    print(f"テーマ：{topic}")

    # 1段目：構成案を作る
    result_step1 = chain_plan.invoke({"topic": topic})
    print("--- 1段目：構成案 ---")
    for number, section in enumerate(result_step1.sections, start=1):
        print(f"{number}. {section}")

    # 2段目：構成案に沿って書く
    outline = "\n".join(f"{number}. {section}" for number, section in enumerate(result_step1.sections, start=1))
    result_step2 = chain_write.invoke({"topic": topic, "outline": outline})
    print("--- 2段目：解説文 ---")
    print(result_step2)


if __name__ == "__main__":
    main()
