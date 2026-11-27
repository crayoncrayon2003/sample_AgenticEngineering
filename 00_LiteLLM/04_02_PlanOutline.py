import os

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


llm = make_llm("low")

#####################################
# LLMのレスポンス形式を定義する（構成案）
#####################################
class Outline(BaseModel):
    #  sections： 説明する項目を、説明する順に並べた一覧
    sections: list[str] = Field(description="説明する項目を、説明する順に3〜5個")


#####################################
# 1段目：構成案（計画）を作る
#####################################
prompt_plan = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは技術解説文のライターです。文章を書く前に、説明する項目と順番を計画してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "テーマ：{topic}"
    )},
]

# LangChain の chain_plan.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_plan(values: dict) -> Outline:
    message = call_llm(llm, format_prompt(prompt_plan, values), response_format=Outline)
    return Outline.model_validate_json(message.content)


#####################################
# 2段目：構成案に沿って書く
#####################################
prompt_write = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは技術解説文のライターです。渡された構成案の順番に沿って、600字程度の解説文を日本語で書いてください。"
        "項目ごとに見出しを付けてください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "テーマ：{topic}\n\n【構成案】\n{outline}"
    )},
]

# LangChain の chain_write.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_write(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_write, values))
    return message.content


def main():
    """
    題材：文章の構成案を作る。その構成案に沿って解説文を書く。
    """

    topic = "AIにおける強化学習の重要性"
    print(f"テーマ：{topic}")

    # 1段目：構成案を作る
    result_step1 = chain_plan({"topic": topic})
    print("--- 1段目：構成案 ---")
    for number, section in enumerate(result_step1.sections, start=1):
        print(f"{number}. {section}")

    # 2段目：構成案に沿って書く
    outline = "\n".join(f"{number}. {section}" for number, section in enumerate(result_step1.sections, start=1))
    result_step2 = chain_write({"topic": topic, "outline": outline})
    print("--- 2段目：解説文 ---")
    print(result_step2)


if __name__ == "__main__":
    main()
