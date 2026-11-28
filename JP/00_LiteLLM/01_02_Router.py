import os
from typing import Literal

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic


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


prompt_route = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": "社内ヘルプデスクの受付です。問い合わせを読み、担当を1つ選んでください。何についての問い合わせか特定できない場合は、unclear を選んでください。"},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": "{request}"},
]


# LangChain の chain_route.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_route(values: dict) -> Route:
    # response_format に Pydantic のクラスを渡すと、その形のJSONで返ってくる
    message = call_llm(llm, format_prompt(prompt_route, values), response_format=Route)
    return Route.model_validate_json(message.content)


#####################################
# 担当ごとの処理
#####################################
def make_chain_handler(role: str):
    prompt_handler = [
        # system : アプリを作った開発者からの指示
        {"role": "system", "content": role + "回答は日本語で3文以内にしてください。"},
        # user   : アプリを使う利用者の発言
        {"role": "user", "content": "{request}"},
    ]

    # LangChain の chain_handler.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
    def chain_handler(values: dict) -> str:
        return call_llm(llm, format_prompt(prompt_handler, values)).content

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
        print("----------------------------------------")
        print(f"問い合わせ  ：{request}")

        # 1段目：担当を決める
        result_step1 = chain_route({"request": request})
        print(f"振り分け先  ：{result_step1.destination}")
        print(f"振り分け理由：{result_step1.reason}")

        # 2段目：決まった担当の処理で回答する
        result_step2 = chain_handlers[result_step1.destination]({"request": request})

        print(f"回答       ：{result_step2}")


if __name__ == "__main__":
    main()
