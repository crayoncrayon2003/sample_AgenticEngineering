import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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
# LLMのレスポンス形式を定義する（目標に対する最終評価）
#####################################
class GoalReview(BaseModel):
    #  feedback ： 目標ごとの評価
    #  goals_met： すべての目標を満たしていれば True
    #  評価 → 判定 の順に並べ、評価を書き出してから判定させる
    feedback: str = Field(description="目標ごとの評価（日本語）。未達成の目標があれば、その理由")
    goals_met: bool = Field(description="最後に実行したコードが、すべての目標を満たす場合だけ true")


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


# 実行の記録（main で初期化する）
run_log = {"dir": "", "round": 0, "last_path": "", "last_result": ""}


#####################################
# ツール：コードを保存して実行し、結果を返す
#####################################
# 04_06_GoalLoop.py では、Pythonの for 文が「作成 → 実行 → 評価」を繰り返した。
# ここでは、エージェントがこのツールを呼んで自分で実行結果を確かめ、必要なら直して再び呼ぶ
@tool
def run_code(code: str) -> str:
    """Pythonスクリプト全体を受け取り、ファイルに保存して実行し、実行結果を返す。

    Args:
        code: 実行するPythonスクリプト全体（説明文やコードフェンスは付けない）
    """
    run_log["round"] += 1
    path = os.path.join(run_log["dir"], f"round_{run_log['round']}.py")
    result = save_and_run(code, path)
    run_log["last_path"] = path
    run_log["last_result"] = result
    print(f"--- 第{run_log['round']}回 ---")
    print(f"実行結果：{result.splitlines()[0]}")
    return result


# system_prompt : エージェントに設定する役割（作成者と評価者を兼ねる）
system_prompt = (
    "あなたはPythonプログラマーです。依頼と目標を満たすPythonスクリプト全体を書き、"
    "必ず run_code で実行して結果を確かめてください。\n"
    "実行に失敗した場合や、目標を満たしていない場合は、コードを直して再び run_code で実行してください。\n"
    "標準ライブラリだけを使ってください。\n"
    "最後に、最後に実行したコードと実行結果を目標ごとに確かめ、評価を返してください。"
    "実行に失敗している場合や、assert が失敗している場合は、目標を満たしていません。"
)
# ツール呼び出しが終わった後に、最終評価を求める指示
review_request = "最後に実行したコードと実行結果を、目標ごとに確かめて評価してください。"

#####################################
# LLMにツールを渡す
#####################################
# bind_tools：LLMに「使えるツールの一覧」を渡す。LLMは必要に応じてツール呼び出しを返す
llm_with_tools = llm.bind_tools([run_code])
llm_review = llm.with_structured_output(GoalReview, method="json_schema")

# MAX_ROUNDS       : run_code を呼べる回数（LangChain の ToolCallLimitMiddleware に当たる）
# MAX_MODEL_CALLS  : LLMを呼べる回数。ツールを呼ばずに考え続ける場合も止める（LangChain の ModelCallLimitMiddleware に当たる）
MAX_MODEL_CALLS = MAX_ROUNDS * 2 + 2


#####################################
# タスク（ワークフローの1ステップ）
#####################################
# LLMを呼ぶ。返事は「回答」か「ツール呼び出しの依頼」のどちらか
@task
def call_model(messages: list) -> AIMessage:
    return llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)


# LLMに依頼されたツールを1つ実行する
@task
def call_tool(tool_call: dict) -> ToolMessage:
    return run_code.invoke(tool_call)


# 最終評価を GoalReview の形式で返させる（LangChain の response_format に当たる）
@task
def review(messages: list) -> GoalReview:
    return llm_review.invoke([SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=review_request)])


#####################################
# エージェント：繰り返しの回数に上限を設ける。上限に達したら、評価を返さずに None を返す
#####################################
@entrypoint()
def agent(request: str) -> GoalReview | None:
    messages = [HumanMessage(content=request)]
    model_calls = 1
    response = call_model(messages).result()

    # LLMがツール呼び出しを返す間は、ツールを実行して、もう一度LLMを呼ぶ
    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if run_log["round"] >= MAX_ROUNDS:
                print(f"  [上限] run_code は{MAX_ROUNDS}回までのため、エージェントを止めます。")
                return None
            tool_messages.append(call_tool(tool_call).result())
        messages = add_messages(messages, [response] + tool_messages)
        if model_calls >= MAX_MODEL_CALLS:
            print(f"  [上限] LLMの呼び出しは{MAX_MODEL_CALLS}回までのため、エージェントを止めます。")
            return None
        model_calls += 1
        response = call_model(messages).result()

    # ツール呼び出しが無くなったら、最終評価を返させる
    return review(add_messages(messages, [response])).result()


def main():
    """
    題材：
        LLMにPythonの関数を作らせる。04_06_GoalLoop.py と同じ依頼と目標を使う。
        01_LangChain/04_06_GoalLoop_Builtin.py と同じ流れで組み立てる。

    04_06_GoalLoop.py との違い：
        04_06_GoalLoop.py ：Pythonの繰り返しが「作成者（LLM）→ 実行 → 評価者（LLM）」を最大5回繰り返す。
        このファイル       ：エージェントが run_code ツールで自分でコードを実行し、結果を見て直す。
                             繰り返しの回数は、run_code の回数とLLMを呼ぶ回数の上限で制限する。
                             評価は、エージェントが最後に GoalReview の形式で返す。

    流れ：
        1. エージェントが、コードを書いて run_code で実行する。
        2. エージェントが、実行結果を見て、目標を満たさなければ直して再び実行する（最大5回）。
        3. エージェントが、最後に目標ごとの評価（GoalReview）を返す。
        4. Pythonが、最後の実行が成功していて、評価がすべての目標を満たすとしていれば採用する。
        各回のコードは、outputs/goal_loop/ に保存する。
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

    # 各回のコードを保存するフォルダー
    run_log.update({
        "dir": os.path.join(GOAL_DIR, datetime.now().strftime("%Y%m%d_%H%M%S") + "_builtin"),
        "round": 0, "last_path": "", "last_result": "",
    })
    os.makedirs(run_log["dir"], exist_ok=True)

    goals_text = "\n".join(f"- {goal}" for goal in goals)
    request = f"【依頼】\n{use_case}\n\n【目標】\n{goals_text}"
    review = agent.invoke(request)

    # 上限に達して止まった場合は、最終評価が無い
    if review is None:
        print(f"回数の上限に達したため、途中で終了しました。記録：{run_log['dir']}")
        return
    print(f"評価：{'目標達成' if review.goals_met else '未達成'}")
    print(f"  {review.feedback}")

    if review.goals_met and run_log["last_result"].startswith("実行成功"):
        print(f"すべての目標を満たしました。採用したコード：{run_log['last_path']}")
    else:
        print(f"目標を達成できませんでした。記録：{run_log['dir']}")


if __name__ == "__main__":
    main()
