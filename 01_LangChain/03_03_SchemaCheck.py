import json
import os

from dotenv import load_dotenv
from jsonschema import Draft202012Validator, FormatChecker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
SCHEMA_PATH = os.path.join(ROOT, "data", "schemas", "user_profile.schema.json")


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
# JSON Schema（出力の形と条件を決めたファイル）を読み込む
#####################################
with open(SCHEMA_PATH, encoding="utf-8") as file:
    schema = json.load(file)
# スキーマそのものが正しい書き方かを確かめる
Draft202012Validator.check_schema(schema)


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
# 2段目：返ってきたJSONを、JSON Schemaで検証する
#####################################
def validate_output(raw_json: str) -> dict:
    # JSONとして読めるか
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as error:
        return {"status": "invalid_json", "errors": [str(error)]}

    # スキーマの条件（必須の項目、型、メールアドレスや日付の形式など）を満たすか
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [f"{error.json_path}：{error.message}" for error in validator.iter_errors(data)]
    if errors:
        return {"status": "schema_mismatch", "errors": errors}
    return {"status": "valid", "errors": []}


def main():
    """
    題材：AIが文章から抜き出したJSONを、あらかじめ決めたJSON Schemaで検証する。
         文章は、合計2件ある。2件目は、メールアドレスと生年月日の形式が正しくない文章。
    """

    print(f"スキーマ：{SCHEMA_PATH}")

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
