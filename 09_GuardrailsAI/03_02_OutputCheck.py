import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from guardrails import Guard
from guardrails.settings import settings
from guardrails.validators import FailResult, PassResult, Validator, register_validator

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
# Guardrails AI の準備
#####################################
# 利用状況の送信（テレメトリ）を止める。設定ファイル（~/.guardrailsrc）が無いと、送信が有効になるため。
#   Validator は作るときの設定を使うため、Validator を作る前に無効にする。Guard は configure でも無効にする
settings.rc.enable_metrics = False


#####################################
# 2段目：作った文章を、Guardrails AI の Guard で検査する
#####################################
# Validator：1つの検査。_validate で、合格なら PassResult、不合格なら FailResult（理由つき）を返す。
# register_validator で名前を付けて登録すると、Guard に組み込める
@register_validator(name="sample/not-empty", data_type="string")
class NotEmpty(Validator):
    def _validate(self, value, metadata):
        if not value.strip():
            return FailResult(error_message="回答が空です。")
        return PassResult()


@register_validator(name="sample/max-length", data_type="string")
class MaxLength(Validator):
    def __init__(self, max_length: int, on_fail=None):
        super().__init__(on_fail=on_fail, max_length=max_length)
        self.max_length = max_length

    def _validate(self, value, metadata):
        if len(value) > self.max_length:
            return FailResult(error_message=f"回答が{self.max_length}文字を超えています。")
        return PassResult()


@register_validator(name="sample/no-email", data_type="string")
class NoEmail(Validator):
    def _validate(self, value, metadata):
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value):
            return FailResult(error_message="メールアドレスが含まれています。")
        return PassResult()


@register_validator(name="sample/no-phone", data_type="string")
class NoPhone(Validator):
    def _validate(self, value, metadata):
        if re.search(r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)", value):
            return FailResult(error_message="電話番号が含まれています。")
        return PassResult()


# Guard：複数の Validator をまとめて実行する。
#   on_fail="noop"：不合格でも文章を書き換えず、結果だけを記録する（ほかに、例外で止める exception、LLMに作り直させる reask などがある）
guard = Guard().use(
    NotEmpty(on_fail="noop"),
    MaxLength(max_length=MAX_ANSWER_LENGTH, on_fail="noop"),
    NoEmail(on_fail="noop"),
    NoPhone(on_fail="noop"),
)
guard.configure(allow_metrics_collection=False)


# 検査で見つかった問題の一覧を返す。問題が無ければ空のリスト
def inspect_output(answer: str) -> list[str]:
    outcome = guard.validate(answer)
    return [summary.failure_reason for summary in outcome.validation_summaries or []]

def main():
    """
    題材：
        AIが作った文章を、利用者に返す前に検査する。問題があれば、返却を保留する。
        検査には、Guardrails AI の Guard と、自作の Validator を使う。

    検査の対象：
        AIが作った文章。次の4つを、それぞれ1つの Validator で検査する。
          1. NotEmpty ：文章が空でないか
          2. MaxLength：MAX_ANSWER_LENGTH（160）文字以内か
          3. NoEmail  ：メールアドレスを含んでいないか
          4. NoPhone  ：電話番号を含んでいないか

    依頼（2件）：
        1. 勉強会が15時に始まることの案内文                    → 問題なし。そのまま返す
        2. 同じ案内文に、架空の連絡先（メールアドレス・電話番号）を含める → 不合格。返却を保留する

    検査すること（inspect_output）：
        Guard が、4つの Validator をまとめて実行する。LLMは使わない。
        Validator は、合格なら PassResult、不合格なら FailResult（理由つき）を返す。
        on_fail="noop" のため、不合格でも文章は書き換えず、不合格の理由だけを集める。

    流れ（依頼1件分）：
        1. 文章を作るチェーンが、依頼から文章を作る。
        2. inspect_output が、文章を Guard で検査し、不合格の理由の一覧を返す。
        3. 理由が無ければ文章を返し、理由があれば理由を表示して、返却を保留する。

    01_LangChain/03_02_OutputCheck.py との違い：
        検査（inspect_output）の中身だけが違う。01_LangChain は検査を1つの関数に書き、こちらは検査ごとに Validator を作り、Guard でまとめる。
        依頼・文章の作り方・流れ・表示する理由は同じ。
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
        result_step1 = chain_answer.invoke({"request": request})
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
