import os
import time

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


#####################################
# 2種類のモデル：軽いモデルと、高性能なモデル
#####################################
# 料金や速度は、提供元・時期によって変わります。最新の情報は、各社の料金ページで確認してください
if PROVIDER == "openai":
    MODEL_NAMES = {"light": "gpt-5-nano", "advanced": "gpt-5-mini"}
else:
    MODEL_NAMES = {"light": "claude-haiku-5-5", "advanced": "claude-sonnet-5-5"}

# 振り分けの基準：質問の文字数（文字数は難しさそのものではなく、教材用の簡易な基準）
CHARACTER_THRESHOLD = 100

# make_llm の設定をもとに、モデル名だけを差し替える（make_llm は、APIキーの確認も行う）
LLMS = {}
for route, model_name in MODEL_NAMES.items():
    LLMS[route] = make_llm("minimal")
    LLMS[route]["model"] = f"{PROVIDER}/{model_name}"

# system_prompt : 回答の役割（2つのモデルで共通）
system_prompt = (
    "あなたはAIエージェントとデータ処理の技術解説係です。"
    "質問に日本語で分かりやすく答えてください。不明な仕様は推測で断定しないでください。"
)


# 質問の文字数で、使うモデルを選ぶ
def select_route(question: str) -> str:
    if len(question) < CHARACTER_THRESHOLD:
        return "light"
    return "advanced"


def main():
    """
    題材：質問の長さで、軽いモデルと高性能なモデルを使い分け、費用と時間を抑える。
         質問は、合計2件ある。1件目は短い質問、2件目は長く複雑な質問。
    """

    print(f"振り分けの基準：{CHARACTER_THRESHOLD}文字未満なら light、それ以上なら advanced")

    # 2件の質問
    questions = [
        "ETLとは何ですか？",
        "複数のCSVファイルを毎日読み込み、列名の違い、文字コードの違い、欠損値、重複データを処理して、集計結果をJSONで保存したいです。途中で一部のファイルの読み込みに失敗した場合も考慮してください。処理の順序、各段階での検証、再試行と途中再開の方法を、設計上の注意点とともに説明してください。",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f"質問（{len(question)}文字）：{question}")

        # 1段目：使うモデルを選ぶ
        route = select_route(question)
        print(f"振り分け先：{route}（{MODEL_NAMES[route]}）")

        # 2段目：選んだモデルのエージェントで答え、時間と使用量を記録する
        start = time.perf_counter()
        response = litellm.completion(
            model=LLMS[route]["model"],
            reasoning_effort=LLMS[route]["reasoning_effort"],
            max_tokens=LLMS[route]["max_tokens"],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            timeout=120,
            num_retries=1,
        )
        elapsed = time.perf_counter() - start

        answer = response.choices[0].message
        # 使用量（トークン数）は、レスポンスの usage に入っている
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
        print(f"時間：{elapsed:.1f}秒、トークン数：入力 {usage.get('input_tokens')}、出力 {usage.get('output_tokens')}")
        print(f"回答：{answer.content}")


if __name__ == "__main__":
    main()
