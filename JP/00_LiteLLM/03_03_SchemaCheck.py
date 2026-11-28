import json
import os

import litellm
from dotenv import load_dotenv
from jsonschema import Draft202012Validator, FormatChecker

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
SCHEMA_PATH = os.path.join(ROOT, "data", "schemas", "user_profile.schema.json")


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
# JSON Schema（出力の形と条件を決めたファイル）を読み込む
#####################################
with open(SCHEMA_PATH, encoding="utf-8") as file:
    schema = json.load(file)
# スキーマそのものが正しい書き方かを確かめる
Draft202012Validator.check_schema(schema)


#####################################
# 1段目：文章から利用者情報をJSONで抜き出す
#####################################
prompt_extract = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは文章から利用者情報を抜き出す係です。"
        "次のJSON Schemaに対応するJSONオブジェクトだけを返してください。説明文やコードフェンスは付けないでください。"
        "文章に無い値を推測して追加しないでください。"
        "原文に書かれたメールアドレスや生年月日は、形式が正しくなくても直さず、そのまま入れてください。"
        "正しいかどうかは、後でPythonが検証します。\n\n"
        "JSON Schema：\n{schema}"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "抜き出す対象の文章：\n{text}"
    )},
]

# LangChain の chain_extract.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_extract(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_extract, values))
    return message.content



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
        result_step1 = chain_extract({"schema": json.dumps(schema, ensure_ascii=False), "text": text})
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
