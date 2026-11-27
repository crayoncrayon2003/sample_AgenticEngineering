import json
import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
GOAL_DIR = os.path.join(ROOT, "outputs", "goal_loop")


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


MAX_ROUNDS = 5

llm = make_llm("low")

#####################################
# LLMのレスポンス形式を定義する（目標に対する評価）
#####################################
class GoalReview(BaseModel):
    #  feedback ： 目標ごとの評価と、未達成なら次に直すべき点
    #  goals_met： すべての目標を満たしていれば True
    #  評価 → 判定 の順に並べ、評価を書き出してから判定させる
    feedback: str = Field(description="目標ごとの評価と、未達成の場合に直すべき具体的な点")
    goals_met: bool = Field(description="すべての目標を満たす場合だけ true")


#####################################
# 作成者：目標を満たすコードを書くチェーン
#####################################
prompt_write = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたはPythonプログラマーです。依頼と目標を満たすPythonスクリプト全体を書いてください。"
        "前回のコード・実行結果・評価がある場合は、それを踏まえて直してください。"
        "標準ライブラリだけを使い、説明文やコードフェンスを付けず、Pythonコードだけを出力してください。"
    ),(
        # user   : 依頼・目標・前回の結果
        "user",
        "【依頼】\n{use_case}\n\n【目標】\n{goals}\n\n【前回のコード】\n{code}\n\n"
        "【前回の実行結果】\n{run_result}\n\n【前回の評価】\n{feedback}"
    ),
])
chain_write = prompt_write | llm | StrOutputParser()

#####################################
# 評価者：コードと実行結果を、目標に照らして評価するチェーン
#####################################
prompt_review = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたはコードの評価係です。目標ごとに、コードと実際の実行結果を確認してください。"
        "実行に失敗している場合や、assert が失敗している場合は、目標を満たしていません。"
    ),(
        # user   : 評価の対象
        "user",
        "【依頼】\n{use_case}\n\n【目標】\n{goals}\n\n【コード】\n{code}\n\n【実行結果】\n{run_result}"
    ),
])
chain_review = prompt_review | llm.with_structured_output(GoalReview, method="json_schema")


#####################################
# 監視：生成したコードを保存して、実際に実行する
#####################################
# 注意：生成されたコードを、このPCで実際に実行します。
#      安全策は、学習用の簡易なもの（別のプロセスで実行する、APIキーなどの環境変数を渡さない、10秒で打ち切る）です。
def save_and_run(code: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as file:
        file.write(code + "\n")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", path],
            env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "実行失敗：10秒以内に終わりませんでした。"
    if completed.returncode != 0:
        return f"実行失敗（終了コード {completed.returncode}）：\n{completed.stderr.strip()[-800:]}"
    return f"実行成功：\n{completed.stdout.strip()[-800:]}"

#####################################
# タスク（ワークフローの1ステップ）
#####################################
@task
def write_code(use_case: str, goals: str, code: str, run_result: str, feedback: str) -> str:
    return chain_write.invoke({"use_case": use_case, "goals": goals, "code": code, "run_result": run_result, "feedback": feedback})


@task
def review(use_case: str, goals: str, code: str, run_result: str) -> GoalReview:
    return chain_review.invoke({"use_case": use_case, "goals": goals, "code": code, "run_result": run_result})


#####################################
# ワークフロー（繰り返しを、普通の for 文で書く）
#####################################
# inputs：依頼（use_case）、目標（goals）、記録を保存するフォルダー（run_dir）
@entrypoint()
def workflow(inputs: dict) -> dict:
    code = "（なし）"
    run_result = "（なし）"
    feedback = "（なし）"
    for round_no in range(1, MAX_ROUNDS + 1):
        print(f"--- 第{round_no}回 ---")

        # Step1 コードを作成する（2回目以降は、実行結果と評価を踏まえて直す）
        code = write_code(inputs["use_case"], inputs["goals"], code, run_result, feedback).result()

        # Step2 監視：コードを保存して実行する
        code_path = os.path.join(inputs["run_dir"], f"round_{round_no}.py")
        run_result = save_and_run(code, code_path)
        print(f"実行結果：{run_result.splitlines()[0]}")

        # Step3 評価：目標に照らして評価し、記録を保存する
        result_review = review(inputs["use_case"], inputs["goals"], code, run_result).result()
        feedback = result_review.feedback
        print(f"評価：{'目標達成' if result_review.goals_met else '未達成'}")
        print(f"  {feedback}")
        with open(os.path.join(inputs["run_dir"], f"round_{round_no}_review.json"), "w", encoding="utf-8") as file:
            json.dump({"run_result": run_result, "feedback": feedback, "goals_met": result_review.goals_met}, file, ensure_ascii=False, indent=2)

        if result_review.goals_met and run_result.startswith("実行成功"):
            return {"message": f"すべての目標を満たしました。採用したコード：{code_path}"}

    return {"message": f"{MAX_ROUNDS}回で目標を達成できませんでした。記録：{inputs['run_dir']}"}


def main():
    """
    題材：
        LLMにPythonの関数を作らせる。あらかじめ「達成すべき目標」を一覧で決めておき、
        作ったコードを実際に動かして確かめ、すべての目標を満たすまで作り直させる。

    登場人物：
        作成者（LLM）：依頼と目標を読んで、Pythonのコードを書く。2回目以降は、前回の実行結果と評価を読んで直す。
        評価者（LLM）：コードと実行結果を読み、目標ごとに満たしているかを判定する。
        Python      ：作成者が書いたコードをファイルに保存し、実際に実行して、結果を評価者に渡す。

    前提：
        依頼：関数 binary_gap(n) を作る。
              正の整数 n を2進数で表したとき、1と1に挟まれた0が最も長く続く個数を返す関数。
              例：9 は2進数で 1001 なので、1と1の間の0は2個 → 2 を返す。
        目標：次の4つをすべて満たすこと。
              1. 正の整数を受け取り、定義どおりの結果を返す。
              2. 整数以外（boolを含む）は TypeError、0以下の整数は ValueError にする。
              3. 9→2、529→4、20→1、15→0、32→0、1→0 を確かめる assert を含める。
              4. if __name__ == '__main__': の中で、使用例と結果を表示する。

    流れ（1回分。最大5回まで繰り返す）：
        1. 作成者が、コードを書く（2回目以降は、前回の実行結果と評価を踏まえて直す）。
        2. Pythonが、コードをファイルに保存して実行する。assert が失敗すると「実行失敗」になる。
        3. 評価者が、コードと実行結果を読み、目標ごとに満たしているかを判定する。
        4. 実行に成功し、評価者がすべての目標を満たすと判定したら終了する。そうでなければ 1 に戻る。
        各回のコード・実行結果・評価は、outputs/goal_loop/ に保存する。
    """

    use_case = (
        "正の整数nを受け取り、2進数で表したときに1と1に挟まれた連続する0の最大個数を返す関数 binary_gap(n) を作成してください。"
        "該当する0の並びが無い場合は0を返してください。"
    )
    goals = [
        "正の整数を受け取り、定義どおりの結果を返す。",
        "整数以外（boolを含む）は TypeError、0以下の整数は ValueError にする。",
        "9→2、529→4、20→1、15→0、32→0、1→0 を確かめる assert を含める。",
        "if __name__ == '__main__': の中で、使用例と結果を表示する。",
    ]
    print("目標：")
    for goal in goals:
        print(f"  ・{goal}")

    # 各回の記録（コード・実行結果・評価）を保存するフォルダー
    run_dir = os.path.join(GOAL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # 作成 → 実行 → 評価 を、すべての目標を満たすか上限回数に達するまで繰り返す
    inputs = {"use_case": use_case, "goals": "\n".join(f"- {goal}" for goal in goals), "run_dir": run_dir}
    result = workflow.invoke(inputs)
    print(result["message"])


if __name__ == "__main__":
    main()
