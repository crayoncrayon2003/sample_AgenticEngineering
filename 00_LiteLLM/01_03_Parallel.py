import os
import time
from concurrent.futures import ThreadPoolExecutor

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


llm = make_llm("minimal")

#####################################
# 1段目：タスク
#####################################
# 1つのタスクを生成する関数
def make_chain_task(instruction: str):
    prompt_task = [
        # system : アプリを作った開発者からの指示
        {"role": "system", "content": instruction + "日本語で簡潔に出力してください。"},
        # user   : アプリを使う利用者の発言
        {"role": "user", "content": "テーマ：{topic}"},
    ]

    # LangChain の chain_task.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
    def chain_task(values: dict) -> str:
        return call_llm(llm, format_prompt(prompt_task, values)).content

    return chain_task

# 3つのタスク「要約」「よくある誤解」「用語」を作る
chain_summary        = make_chain_task("テーマを3文で要約してください。")
chain_misconceptions = make_chain_task("テーマについて、よくある誤解を2つ挙げてください。")
chain_terms          = make_chain_task("テーマの重要な用語を5つ、1行ずつ短い説明付きで挙げてください。")


# 3つのタスクを同時に実行する（LangChain の RunnableParallel に当たる）
def chain_parallel(values: dict) -> dict:
    # ThreadPoolExecutor：処理を別々のスレッドで同時に動かす（Python標準ライブラリ）
    with ThreadPoolExecutor() as executor:
        future_summary        = executor.submit(chain_summary, values)
        future_misconceptions = executor.submit(chain_misconceptions, values)
        future_terms          = executor.submit(chain_terms, values)
        return {
            "summary": future_summary.result(),
            "misconceptions": future_misconceptions.result(),
            "terms": future_terms.result(),
        }

#####################################
# 2段目：結果をまとめる処理
#####################################
prompt_merge = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "渡された要約・誤解・用語を使い、初学者向けの解説文を日本語で書いてください。"
        "誤解は正しい説明と対にしてください。400字程度にしてください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "テーマ：{topic}\n\n【要約】\n{summary}\n\n"
        "【よくある誤解】\n{misconceptions}\n\n【用語】\n{terms}"
    )},
]


# LangChain の chain_merge.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_merge(values: dict) -> str:
    return call_llm(llm, format_prompt(prompt_merge, values)).content


#####################################
# 2つをつないだ処理
#####################################
# 1段目の結果を、そのまま2段目へ渡す
def chain_all(values: dict) -> str:
    result = chain_parallel(values)
    return chain_merge({**values, **result})


def main():
    """
    題材：あるテーマについて「要約」「よくある誤解」「用語」を並列実行する。実行結果を解説文に統合する。
    """

    topic = "AIエージェント"
    print(f"テーマ：{topic}")

    # ##################################################################
    # 方法1：順番に呼び出し、途中の結果を確認する
    # ##################################################################
    start = time.perf_counter()
    result_step1 = chain_parallel({"topic": topic})
    print(f"--- 1段目：並列実行（{time.perf_counter() - start:.1f}秒）---")
    print("[summary]")
    print(result_step1["summary"])
    print("[misconceptions]")
    print(result_step1["misconceptions"])
    print("[terms]")
    print(result_step1["terms"])

    result_step2 = chain_merge({"topic": topic, **result_step1})
    print("--- 2段目：統合した解説文 ---")
    print(result_step2)

    # 比較のため、1段目の3つの処理を順番に実行する
    start = time.perf_counter()
    chain_summary({"topic": topic})
    chain_misconceptions({"topic": topic})
    chain_terms({"topic": topic})
    print(f"（参考）1段目を順番に実行した場合：{time.perf_counter() - start:.1f}秒")

    # ##################################################################
    # 方法2：2つをつないだ処理として、並列実行から統合までを行う
    # ##################################################################
    print("--- 2つをつないだ処理で実行 ---")
    print(chain_all({"topic": topic}))


if __name__ == "__main__":
    main()
