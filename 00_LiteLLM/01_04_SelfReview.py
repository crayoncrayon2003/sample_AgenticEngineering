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


llm = make_llm("minimal")

#####################################
# LLMのレスポンス形式を定義する（評価結果）
#####################################
class Review(BaseModel):
    #  passed： 要件をすべて満たし、修正が不要なら True。
    #  issues： 修正すべき点の一覧。合格なら空のリスト。
    passed: bool      = Field(description="要件をすべて満たし、修正不要ならtrue")
    issues: list[str] = Field(description="修正すべき点。合格なら空のリスト")


#####################################
# 作成者（初回の作成と、指摘を受けた修正を兼ねる）
#####################################
prompt_write = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたはPythonプログラマーです。課題を満たすコードを書いてください。"
        "前回のコードと指摘がある場合は、指摘に対応した修正後のコード全体を書いてください。"
        "説明文やMarkdownのコードフェンスを付けず、Pythonコードだけを出力してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": "【課題】\n{task}\n\n【前回のコード】\n{code}\n\n【指摘】\n{issues}"},
]


# LangChain の chain_write.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_write(values: dict) -> str:
    return call_llm(llm, format_prompt(prompt_write, values)).content


#####################################
# 評価者
#####################################
prompt_review = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたはコードレビュー担当です。課題の要件、バグ、境界値、docstringを確認してください。"
        "コードは実行できないため、読んで判断してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": "【課題】\n{task}\n\n【コード】\n{code}"},
]


# LangChain の chain_review.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_review(values: dict) -> Review:
    message = call_llm(llm, format_prompt(prompt_review, values), response_format=Review)
    return Review.model_validate_json(message.content)


#####################################
# 作成 → 評価 を繰り返す
#####################################
def self_review(task: str, MAX_ROUNDS : int) -> str:
    # code  ：その時点の最新版のコード（作成するたびに更新する）
    # issues：評価者からの指摘（評価するたびに更新する）
    # 初回は前回のコードも指摘も無いため、「（なし）」を渡す
    code = "（なし）"
    issues = "（なし）"

    for round_no in range(1, MAX_ROUNDS + 1):
        # Step1 コードを作成する（2回目以降は、指摘に沿って修正する）
        result_write = chain_write({"task": task, "code": code, "issues": issues})
        code = result_write
        print(f"--- 第{round_no}版 ---")
        print(code)

        # Step2 コードを評価する
        result_review = chain_review({"task": task, "code": code})
        if result_review.passed:
            print("評価：合格")
            return code

        print("評価：要修正")
        for issue in result_review.issues:
            print(f"  - {issue}")
        issues = "\n".join(result_review.issues)

    print("上限回数に達したため、最後の版で終了します。")
    return code


def main():
    """
    題材：Pythonの関数を書かせて、別のプロンプトで評価し、指摘に沿って直す。
         合格するか、MAX_ROUNDS 回に達するまで繰り返す。
    """

    # 課題（作成する関数の要件）
    task = (
        "format_duration という関数を作成してください。\n"
        "1. 秒数の整数 seconds を受け取り、「1日2時間3分4秒」の形の文字列を返す。\n"
        "2. 0になる単位は省略する（例：3600 → 「1時間」、90061 → 「1日1時間1分1秒」）。\n"
        "3. seconds が0なら「0秒」を返す。\n"
        "4. seconds が int でない場合は TypeError を発生させる。bool も int ではないものとして扱う。\n"
        "5. seconds が負なら ValueError を発生させる。\n"
        "6. docstring は日本語で書き、doctest形式の使用例を3つ以上含める。"
    )
    # 作成 → 評価 を繰り返し、最終版のコードを受け取る
    final_code = self_review(task, 3)
    print("--- 最終コード ---")
    print(final_code)


if __name__ == "__main__":
    main()
