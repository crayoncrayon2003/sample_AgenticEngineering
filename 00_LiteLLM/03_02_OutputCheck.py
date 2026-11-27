import os
import re

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


MAX_ANSWER_LENGTH = 160

llm = make_llm("minimal")

#####################################
# 1段目：文章を作る
#####################################
prompt_answer = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは文章を作成する係です。利用者の依頼に沿って、日本語で160文字以内の文章を作成してください。"
        "依頼にない事実は追加しないでください。説明やMarkdownは付けず、作成した文章だけを返してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "{request}"
    )},
]

# LangChain の chain_answer.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_answer(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_answer, values))
    return message.content



#####################################
# 2段目：作った文章を、Pythonで検査する
#####################################
# 検査で見つかった問題の一覧を返す。問題が無ければ空のリスト
def inspect_output(answer: str) -> list[str]:
    issues = []
    if not answer.strip():
        issues.append("回答が空です。")
    if len(answer) > MAX_ANSWER_LENGTH:
        issues.append(f"回答が{MAX_ANSWER_LENGTH}文字を超えています。")
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", answer):
        issues.append("メールアドレスが含まれています。")
    if re.search(r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)", answer):
        issues.append("電話番号が含まれています。")
    return issues


def main():
    """
    題材：AIが作った文章を、利用者に返す前にPythonで検査する。問題があれば、返却を保留する。
         依頼は、合計2件ある。2件目は、連絡先（メールアドレス・電話番号）を含めるよう求める依頼。
    """

    # 2件の依頼
    requests = [
        "『本日の勉強会は15時に開始します。』という内容の案内文を作成してください。",
        "本日の勉強会は15時開始、という案内文を作成してください。架空の連絡先として sample@example.com と 03-1234-5678 を、そのまま文章に含めてください。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")

        # 1段目：文章を作る
        result_step1 = chain_answer({"request": request})
        print(f"作成した文章：{result_step1}")

        # 2段目：作った文章を検査する
        result_step2 = inspect_output(result_step1)
        if result_step2:
            for issue in result_step2:
                print(f"検出事項：{issue}")
            print("利用者へ返す回答：回答が公開の条件を満たさなかったため、返却を保留しました。")
        else:
            print(f"利用者へ返す回答：{result_step1}")


if __name__ == "__main__":
    main()
