import datetime
import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from guardrails import Guard
from guardrails.settings import settings
from guardrails.validators import FailResult, PassResult, Validator, register_validator
from pydantic import BaseModel, ConfigDict, Field

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
# Guardrails AI の準備
#####################################
# 利用状況の送信（テレメトリ）を止める。設定ファイル（~/.guardrailsrc）が無いと、送信が有効になるため。
#   Validator は作るときの設定を使うため、Validator を作る前に無効にする。Guard は configure でも無効にする
settings.rc.enable_metrics = False


#####################################
# 出力の形と条件を、Pydantic のクラスで決める
#####################################
# 01_LangChain/03_03_SchemaCheck.py の JSON Schema（data/schemas/user_profile.schema.json）と同じ条件を、Pydantic で書く
class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 決めていない項目があれば不合格にする（additionalProperties: false）
    name: str = Field(min_length=1, description="利用者の氏名。")
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="利用者のメールアドレス。")
    date_of_birth: Optional[str] = Field(default=None, description="生年月日。文字列の場合はYYYY-MM-DD形式。")
    interests: list[str] = Field(default_factory=list, description="興味・関心のある分野の一覧。")


# LLMに渡すスキーマ（JSON Schema の形）。Pydantic のクラスから作る
schema = UserProfile.model_json_schema()


# Validator：型や形式だけでは確かめられない条件を、Pythonで書いて検査する。
# 「1995-02-30」のように、形式は合っていても存在しない日付は、ここで不合格にする
@register_validator(name="sample/existing-date", data_type="string")
class ExistingDate(Validator):
    def _validate(self, value, metadata):
        if value is None:
            return PassResult()
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            return FailResult(error_message=f"存在する日付（YYYY-MM-DD）ではありません：{value}")
        return PassResult()


# Guard.for_pydantic：Pydantic のクラスの条件で、LLMの出力を検査する Guard を作る。
# use(..., on="$.date_of_birth")：指定した項目だけに、Validator を追加する
guard = Guard.for_pydantic(UserProfile).use(ExistingDate(on_fail="noop"), on="$.date_of_birth")
guard.configure(allow_metrics_collection=False)


#####################################
# 1段目：文章から利用者情報をJSONで抜き出すチェーン
#####################################
prompt_extract = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは文章から利用者情報を抜き出す係です。"
        "次のJSON Schemaに対応するJSONオブジェクトだけを返してください。説明文やコードフェンスは付けないでください。"
        "文章に無い値を推測して追加しないでください。"
        "原文に書かれたメールアドレスや生年月日は、形式が正しくなくても直さず、そのまま入れてください。"
        "正しいかどうかは、後でPythonが検証します。\n\n"
        "JSON Schema：\n{schema}"
    ),(
        # user   : アプリを使う利用者の発言
        "user",
        "抜き出す対象の文章：\n{text}"
    ),
])
chain_extract = prompt_extract | llm | StrOutputParser()


#####################################
# 2段目：返ってきたJSONを、Guard で検証する
#####################################
def validate_output(raw_json: str) -> dict:
    # parse：LLMの出力を読み込み、Pydantic のクラスの条件と Validator で検査する。num_reasks=0：作り直しをLLMに頼まない
    outcome = guard.parse(raw_json, num_reasks=0)

    # JSONとして読めるか
    if outcome.error:
        return {"status": "invalid_json", "errors": [outcome.error]}

    # クラスの条件（必須の項目、型、メールアドレスの形式など）と、Validator（存在する日付か）を満たすか
    errors = [f"{summary.property_path}：{summary.failure_reason}" for summary in outcome.validation_summaries or []]
    if outcome.reask is not None:
        errors += [result.error_message for result in outcome.reask.fail_results]
    if not outcome.validation_passed:
        return {"status": "schema_mismatch", "errors": errors}
    return {"status": "valid", "errors": []}

def main():
    """
    題材：
        AIが文章から抜き出した利用者情報（JSON）を、あらかじめ決めた条件で検証する。条件を満たさなければ、後続の処理に渡さない。
        検証には、Guardrails AI の Guard と、Pydantic のクラス（UserProfile）を使う。

    検証の対象：
        AIが返したJSON。次の条件を満たすかを検証する。
          1. UserProfile の条件：name は1文字以上、email はメールアドレスの形、決めていない項目を含まない など
          2. ExistingDate     ：date_of_birth が、存在する日付（YYYY-MM-DD）か

    文章（2件）：
        1. 山田花子さんの情報（正しい形式）                  → 検証に合格する
        2. 山田太郎さんの情報（生年月日が存在しない 1995-02-30） → 検証に失敗する

    検証すること（validate_output）：
        guard.parse が、JSONを読み込み、UserProfile の条件と ExistingDate で検証する。LLMは使わない。
          - JSONとして読めない      ：invalid_json
          - 条件を満たさない        ：schema_mismatch（満たさなかった条件の一覧を返す）
          - すべての条件を満たす    ：valid
        「1995-02-30」は、形（YYYY-MM-DD）は合っているため、UserProfile の条件だけでは見つけられない。そのため Validator で検査する。

    流れ（文章1件分）：
        1. 抜き出すチェーンが、文章から利用者情報をJSONで抜き出す。
        2. validate_output が、JSONを Guard で検証する。
        3. 検証結果を表示する。合格なら後続の処理に渡せる。

    01_LangChain/03_03_SchemaCheck.py との違い：
        条件の書き方と検証の中身が違う。01_LangChain は条件を JSON Schema のファイルに書き、jsonschema で検証する。
        こちらは条件を Pydantic のクラスに書き、Guard で検証する。存在しない日付は、自作の Validator で検査する。
        文章・抜き出し方・流れは同じ。
    """

    print(f"スキーマ：{UserProfile.__name__}")

    # 2件の文章
    texts = [
        "名前は山田花子です。メールアドレスは hanako@example.com です。生年月日は1995-07-21です。興味のある分野はPythonとデータ分析です。",
        "名前は山田太郎です。メールアドレスとして taro@example.com が記録されています。生年月日として1995-02-30が記録されています。興味のある分野は機械学習です。",
    ]
    for text in texts:
        print("----------------------------------------")
        print(f"文章：{text}")

        # 1段目：文章から利用者情報をJSONで抜き出す
        result_step1 = chain_extract.invoke({"schema": json.dumps(schema, ensure_ascii=False), "text": text})
        print(f"AIが返したJSON：{result_step1}")

        # 2段目：JSON Schemaで検証する
        result_step2 = validate_output(result_step1)
        print(f"検証結果：{result_step2['status']}")
        for error in result_step2["errors"]:
            print(f"  - {error}")
        if result_step2["status"] == "valid":
            print("スキーマに適合しています。後続の処理に渡せます。")
        else:
            print("検証に失敗しました。後続の処理には渡しません。")


if __name__ == "__main__":
    main()
