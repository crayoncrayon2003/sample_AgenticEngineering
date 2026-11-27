import json
import os

import litellm
from dotenv import load_dotenv

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
# 1段目：レビューから要点を抜き出す
#####################################
prompt_extract = [
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "次の商品レビューから、良い点と気になる点を箇条書きで抜き出してください。\n"
        "レビューに書かれていないことは追加しないでください。\n\n"
        "{review}"
    )},
]


# LangChain の chain_extract.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_extract(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_extract, values))
    return message.content


#####################################
# 2段目：要点をJSONに整える
#####################################
prompt_json = [
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "次の要点を、キーが good（配列）と concerns（配列）のJSONに変換してください。\n"
        "JSONだけを出力してください。\n\n"
        "{points}"
    )},
]


# LangChain の chain_json.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_json(values: dict) -> dict:
    # response_format：JSONで返すように指定する。返ってきた文字列を辞書に変換する
    message = call_llm(llm, format_prompt(prompt_json, values), response_format={"type": "json_object"})
    return json.loads(message.content)


#####################################
# 2つをつないだ処理
#####################################
# 1段目の出力を、2段目の {points} に入れる
def chain_all(values: dict) -> dict:
    return chain_json({"points": chain_extract(values)})


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
    # Step1 良い点・気になる点を抜き出す
    result_step1 = chain_extract({"review": review})
    print("--- 1段目：抜き出した要点 ---")
    print(result_step1)

    # Step2 jsonに変換する
    result_step2 = chain_json({"points": result_step1})
    print("--- 2段目：JSON ---")
    print(result_step2)

    # ##################################################################
    # 方法2：2つをつないだ処理として、良い点・気になる点を抜き出しJSONに変換する
    # ##################################################################
    print("--- 2つをつないだ処理で実行 ---")
    print(chain_all({"review": review}))


if __name__ == "__main__":
    main()
