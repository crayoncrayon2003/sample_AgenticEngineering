import os
from typing import Literal

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


llm = make_llm("minimal")

#####################################
# LLMのレスポンス形式を定義する
#####################################
class Route(BaseModel):
    #  reason     ： 振り分け先の担当者を選択した理由。
    #  destination： 振り分け先の担当者を設定する。選択肢は、"expense", "it", "unclear"のいずれかである。
    #  理由（reason） → 担当（destination） の順に並べる。
    #  LLMは前の項目から順に出力するため、先に理由を書かせると、その理由に沿った担当を選びやすくなる。
    reason: str = Field(description="問い合わせの内容と、どの担当が適切かの判断（日本語で1文）")
    destination: Literal["expense", "it", "unclear"] = Field(description="expense：経費精算、it：PCやネットワークの不具合、unclear：判断できない")


prompt_route = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "社内ヘルプデスクの受付です。問い合わせを読み、担当を1つ選んでください。何についての問い合わせか特定できない場合は、unclear を選んでください。",
    ),(
        # user   : アプリを使う利用者の発言
        "user", 
        "{request}"
    ),
])
chain_route = prompt_route | llm.with_structured_output(Route, method="json_schema")

#####################################
# 担当ごとのチェーン
#####################################
def make_chain_handler(role: str):
    prompt_handler = ChatPromptTemplate.from_messages([
        (
            # system : アプリを作った開発者からの指示
            "system",
            role + "回答は日本語で3文以内にしてください。"
        ),(
            # user   : アプリを使う利用者の発言
            "user", 
            "{request}"
        ),
    ])
    chain_handler = prompt_handler | llm | StrOutputParser()
    return chain_handler


chain_handlers = {
    "expense": make_chain_handler("あなたは経理担当です。経費精算の手順を案内してください。"
                                  "社内規程が分からない点は、経理部への確認を勧めてください。"),
    "it"     : make_chain_handler("あなたはIT担当です。利用者が試せる確認手順を案内してください。"),
    "unclear": make_chain_handler("あなたは受付担当です。依頼の内容が分からないため、"
                                  "何に困っているのかを具体的に尋ねてください。"),
}


def main():
    """
    題材：社内ヘルプデスクへの問い合わせを、経費・IT・不明の3つに振り分ける。
         問い合わせは、合計3件ある。
    """

    requests = [
        "出張のタクシー代を精算したいのですが、領収書をなくしました。",
        "会議室のWi-Fiにつながりません。",
        "例の件、どうなりましたか？",
    ]

    for request in requests:
        print()
        print("----------------------------------------")
        print(f"問い合わせ  ：{request}")

        # 1段目：担当を決める
        result_step1 = chain_route.invoke({"request": request})
        print(f"振り分け先  ：{result_step1.destination}")
        print(f"振り分け理由：{result_step1.reason}")

        # 2段目：決まった担当のチェーンで回答する
        result_step2 = chain_handlers[result_step1.destination].invoke({"request": request})

        print(f"回答       ：{result_step2}")


if __name__ == "__main__":
    main()
