import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# NeMo Guardrails の利用状況の送信（テレメトリ）を止める。NeMo Guardrails を import する前に設定する
os.environ.setdefault("NEMO_GUARDRAILS_NO_USAGE_STATS", "1")
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus

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
# 2段目：作った文章を、出力のレール（NeMo Guardrails）で検査する
#####################################
# NeMo Guardrails では、検査の仕組み（レール）を設定（YAML）で組み立てる。
#   rails.output.flows   ：出力を検査するレール。regex check output は、決めたパターン（正規表現）に一致する出力を止める
#   regex_detection      ：止めるパターン。メールアドレス・電話番号・MAX_ANSWER_LENGTH 文字を超える文章
# 設定は、ファイル（config.yml）に書いて RailsConfig.from_path で読み込むこともできる
RAILS_CONFIG = r"""
rails:
  config:
    regex_detection:
      output:
        patterns:
          - '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
          - '(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)'
          - '^[\s\S]{MIN_LENGTH,}$'
  output:
    flows:
      - regex check output
""".replace("MIN_LENGTH", str(MAX_ANSWER_LENGTH + 1))
rails = LLMRails(RailsConfig.from_content(yaml_content=RAILS_CONFIG), llm=llm)


# 検査で見つかった問題の一覧を返す。問題が無ければ空のリスト
# check は、出力のレールだけを実行し、結果（PASSED・BLOCKED）を返す。どのパターンに一致したかまでは返さない
def inspect_output(answer: str) -> list[str]:
    if not answer.strip():
        return ["回答が空です。"]
    result = rails.check([{"role": "user", "content": ""}, {"role": "assistant", "content": answer}])
    if result.status == RailStatus.BLOCKED:
        return [f"レール「{result.rail}」が止めました（メールアドレス・電話番号・{MAX_ANSWER_LENGTH}文字超えのいずれか）。"]
    return []

def main():
    """
    題材：
        AIが作った文章を、利用者に返す前に検査する。問題があれば、返却を保留する。
        検査には、NeMo Guardrails の出力のレールを使う。

    検査の対象：
        AIが作った文章。次の3つを含んでいたら止める（RAILS_CONFIG の regex_detection に、正規表現で書く）。
          1. メールアドレス
          2. 電話番号
          3. MAX_ANSWER_LENGTH（160）文字を超える文章

    依頼（2件）：
        1. 勉強会が15時に始まることの案内文                    → 問題なし。そのまま返す
        2. 同じ案内文に、架空の連絡先（メールアドレス・電話番号）を含める → 止める。返却を保留する

    検査すること（inspect_output）：
        出力のレール regex check output が、文章が正規表現のどれかに一致するかを調べる。LLMは使わない。
        rails.check は、出力のレールだけを実行し、結果（passed：問題なし、blocked：止めた）を返す。
        空の文章かどうかは、レールではなく Python で確かめる。

    流れ（依頼1件分）：
        1. 文章を作るチェーンが、依頼から文章を作る。
        2. inspect_output が、文章を出力のレールで検査する。
        3. 問題が無ければ文章を返し、問題があれば返却を保留する。

    01_LangChain/03_02_OutputCheck.py との違い：
        検査（inspect_output）の中身だけが違う。01_LangChain は正規表現の検査を Python で書き、こちらは NeMo Guardrails のレールで検査する。
        依頼・文章の作り方・流れは同じ。

    注意：
        regex check output は、どのパターン（メールアドレス・電話番号・文字数）に一致したかまでは返さない。
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
