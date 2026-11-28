import os
import time

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

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
# 1段目：タスク
#####################################
# 1つのタスクを生成する関数
def make_chain_task(instruction: str):
    prompt_task = ChatPromptTemplate.from_messages([
        # system : アプリを作った開発者からの指示
        ("system", instruction + "日本語で簡潔に出力してください。"),
        # user   : アプリを使う利用者の発言
        ("user", "テーマ：{topic}"),
    ])
    chain_task = prompt_task | llm | StrOutputParser()
    return chain_task

# 3つのタスク「要約」「よくある誤解」「用語」を作る
chain_parallel = RunnableParallel(
    summary       = make_chain_task("テーマを3文で要約してください。"),
    misconceptions= make_chain_task("テーマについて、よくある誤解を2つ挙げてください。"),
    terms         = make_chain_task("テーマの重要な用語を5つ、1行ずつ短い説明付きで挙げてください。"),
)

#####################################
# 2段目：結果をまとめる処理
#####################################
prompt_merge = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "渡された要約・誤解・用語を使い、初学者向けの解説文を日本語で書いてください。"
        "誤解は正しい説明と対にしてください。400字程度にしてください。",
    ),
    (
        # user   : アプリを使う利用者の発言
        "user",
        "テーマ：{topic}\n\n【要約】\n{summary}\n\n"
        "【よくある誤解】\n{misconceptions}\n\n【用語】\n{terms}",
    ),
])
chain_merge = prompt_merge | llm | StrOutputParser()

#####################################
# --- 2つをつないだ1本のチェーン ---
#####################################
# 入力 {topic} に1段目の結果を追加し、そのまま2段目へ渡す
chain_all = (
    RunnablePassthrough.assign(results=chain_parallel)
    | (lambda x: {"topic": x["topic"], **x["results"]})
    | chain_merge
)


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
    result_step1 = chain_parallel.invoke({"topic": topic})
    print()
    print(f"--- 1段目：並列実行（{time.perf_counter() - start:.1f}秒）---")
    for key, value in result_step1.items():
        print()
        print(f"[{key}]")
        print(value)

    result_step2 = chain_merge.invoke({"topic": topic, **result_step1})
    print()
    print("--- 2段目：統合した解説文 ---")
    print(result_step2)

    # 比較のため、1段目の3つの処理を順番に実行する
    start = time.perf_counter()
    for chain_task in chain_parallel.steps__.values():
        chain_task.invoke({"topic": topic})
    print()
    print(f"（参考）1段目を順番に実行した場合：{time.perf_counter() - start:.1f}秒")

    # ##################################################################
    # 方法2：1本のチェーンとして、 良い点・気になる点を抜き出しJSONに変換する
    # ##################################################################
    print()
    print("--- 1本のチェーンで実行 ---")
    print(chain_all.invoke({"topic": topic}))


if __name__ == "__main__":
    main()
