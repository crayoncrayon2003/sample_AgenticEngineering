import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIDetectionError, PIIMiddleware

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
# 標準ミドルウェア：AIの回答に個人情報が含まれていたら止める
#####################################
# 03_02_OutputCheck.py では、正規表現で検査する関数（inspect_output）を自分で書いた。
# PIIMiddleware は、個人情報（PII）の種類ごとに検出と対処をまとめて引き受ける。
#   pii_type        : 検出する種類。email・credit_card・ip・mac_address・url は標準で用意されている
#   detector        : 標準に無い種類は、正規表現で検出方法を渡す
#   strategy        : 見つけたときの対処。block（例外で止める）・redact（伏せ字にする）・mask（一部を隠す）・hash
#   apply_to_input  : 利用者の入力を検査するか（今回は、依頼文に連絡先が含まれるため検査しない）
#   apply_to_output : AIの回答を検査するか
pii_email = PIIMiddleware(
    "email",
    strategy="block",
    apply_to_input=False,
    apply_to_output=True,
)
pii_phone = PIIMiddleware(
    "phone",
    detector=r"(?<!\d)0\d{1,3}-\d{1,4}-\d{4}(?!\d)",
    strategy="block",
    apply_to_input=False,
    apply_to_output=True,
)

# 検出した種類を、表示用の日本語に直す
PII_LABELS = {"email": "メールアドレス", "phone": "電話番号"}


#####################################
# エージェント：文章を作り、回答を返す前にミドルウェアが検査する
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[],
    # middleware    : 回答に含まれる個人情報を検査する処理
    middleware=[pii_email, pii_phone],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは文章を作成する係です。利用者の依頼に沿って、日本語で160文字以内の文章を作成してください。"
        "依頼にない事実は追加しないでください。説明やMarkdownは付けず、作成した文章だけを返してください。"
    ),
)


def main():
    """
    題材：AIが作った文章を、利用者に返す前に標準ミドルウェアで検査する。問題があれば、返却を保留する。
         03_02_OutputCheck.py と同じ依頼を使う。2件目は、連絡先（メールアドレス・電話番号）を含めるよう求める依頼。
         文字数の検査は、個人情報の検査ではないため標準ミドルウェアには無い。ここだけはPythonで検査する。
    """

    # 2件の依頼
    requests = [
        "『本日の勉強会は15時に開始します。』という内容の案内文を作成してください。",
        "本日の勉強会は15時開始、という案内文を作成してください。架空の連絡先として sample@example.com と 03-1234-5678 を、そのまま文章に含めてください。",
    ]
    for request in requests:
        print("----------------------------------------")
        print(f"依頼：{request}")

        try:
            # エージェントを実行する。回答に個人情報があると、PIIDetectionError が発生する
            result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        except PIIDetectionError as error:
            # block は、最初に見つかった種類で止まる（メールアドレスと電話番号の両方があっても、1種類だけ報告される）
            print(f"検出事項：{PII_LABELS.get(error.pii_type, error.pii_type)}が含まれています（{len(error.matches)}件）。")
            print("利用者へ返す回答：回答が公開の条件を満たさなかったため、返却を保留しました。")
            continue

        answer = result["messages"][-1].text
        print(f"作成した文章：{answer}")
        if len(answer) > MAX_ANSWER_LENGTH:
            print(f"検出事項：回答が{MAX_ANSWER_LENGTH}文字を超えています。")
            print("利用者へ返す回答：回答が公開の条件を満たさなかったため、返却を保留しました。")
        else:
            print(f"利用者へ返す回答：{answer}")


if __name__ == "__main__":
    main()
