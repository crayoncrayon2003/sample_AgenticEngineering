import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task

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


MAX_ANSWER_LENGTH = 160

llm = make_llm("minimal")

#####################################
# 1段目：文章を作るチェーン
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは文章を作成する係です。利用者の依頼に沿って、日本語で160文字以内の文章を作成してください。"
        "依頼にない事実は追加しないでください。説明やMarkdownは付けず、作成した文章だけを返してください。"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "{request}"
    ),
])
chain_answer = prompt_answer | llm | StrOutputParser()


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

#####################################
# タスク：文章を作る
#####################################
@task
def write_answer(request: str) -> str:
    return chain_answer.invoke({"request": request})


#####################################
# ワークフロー（処理の流れを、普通のPythonで書く）
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # 1段目：文章を作る
    result_step1 = write_answer(request).result()

    # 2段目：作った文章を検査し、問題があれば返却を保留する
    result_step2 = inspect_output(result_step1)
    if result_step2:
        final_answer = "回答が公開の条件を満たさなかったため、返却を保留しました。"
    else:
        final_answer = result_step1
    return {"draft": result_step1, "issues": result_step2, "final_answer": final_answer}


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

        # 1段目（文章を作る）と2段目（検査する）を、まとめて実行する
        result = workflow.invoke(request)
        print(f"作成した文章：{result['draft']}")
        for issue in result["issues"]:
            print(f"検出事項：{issue}")
        print(f"利用者へ返す回答：{result['final_answer']}")


if __name__ == "__main__":
    main()
