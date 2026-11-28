import json
import os
import subprocess
import sys
from datetime import datetime

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
GOAL_DIR = os.path.join(ROOT, "outputs", "goal_loop")


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
# 作成者：目標を満たすコードを書く
#####################################
prompt_write = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたはPythonプログラマーです。依頼と目標を満たすPythonスクリプト全体を書いてください。"
        "前回のコード・実行結果・評価がある場合は、それを踏まえて直してください。"
        "標準ライブラリだけを使い、説明文やコードフェンスを付けず、Pythonコードだけを出力してください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "【依頼】\n{use_case}\n\n【目標】\n{goals}\n\n【前回のコード】\n{code}\n\n"
        "【前回の実行結果】\n{run_result}\n\n【前回の評価】\n{feedback}"
    )},
]

# LangChain の chain_write.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_write(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_write, values))
    return message.content


#####################################
# 評価者：コードと実行結果を、目標に照らして評価する
#####################################
prompt_review = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたはコードの評価係です。目標ごとに、コードと実際の実行結果を確認してください。"
        "実行に失敗している場合や、assert が失敗している場合は、目標を満たしていません。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "【依頼】\n{use_case}\n\n【目標】\n{goals}\n\n【コード】\n{code}\n\n【実行結果】\n{run_result}"
    )},
]

# LangChain の chain_review.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_review(values: dict) -> GoalReview:
    message = call_llm(llm, format_prompt(prompt_review, values), response_format=GoalReview)
    return GoalReview.model_validate_json(message.content)


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

    code = "（なし）"
    run_result = "（なし）"
    feedback = "（なし）"
    goals_text = "\n".join(f"- {goal}" for goal in goals)
    for round_no in range(1, MAX_ROUNDS + 1):
        print(f"--- 第{round_no}回 ---")

        # Step1 コードを作成する（2回目以降は、実行結果と評価を踏まえて直す）
        code = chain_write({"use_case": use_case, "goals": goals_text, "code": code,
                                   "run_result": run_result, "feedback": feedback})

        # Step2 監視：コードを保存して実行する
        code_path = os.path.join(run_dir, f"round_{round_no}.py")
        run_result = save_and_run(code, code_path)
        print(f"実行結果：{run_result.splitlines()[0]}")

        # Step3 評価：目標に照らして評価し、記録を保存する
        result_review = chain_review({"use_case": use_case, "goals": goals_text, "code": code, "run_result": run_result})
        feedback = result_review.feedback
        print(f"評価：{'目標達成' if result_review.goals_met else '未達成'}")
        print(f"  {feedback}")
        with open(os.path.join(run_dir, f"round_{round_no}_review.json"), "w", encoding="utf-8") as file:
            json.dump({"run_result": run_result, "feedback": feedback, "goals_met": result_review.goals_met}, file, ensure_ascii=False, indent=2)

        if result_review.goals_met and run_result.startswith("実行成功"):
            print(f"すべての目標を満たしました。採用したコード：{code_path}")
            return

    print(f"{MAX_ROUNDS}回で目標を達成できませんでした。記録：{run_dir}")


if __name__ == "__main__":
    main()
