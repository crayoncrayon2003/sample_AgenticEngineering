import json
import os
import subprocess
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
SCRUM_DIR = os.path.join(ROOT, "outputs", "scrum")


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


llm = make_llm("low")

#####################################
# プロダクトの目標と、バックログ（作る機能の一覧）
#####################################
PRODUCT_GOAL = "売上データを、安全に集計できる小さなコマンドラインプログラムを作る。"
BACKLOG = [
    {"id": "PBI-01", "value": "正常な売上の合計と行数を取得できる。", "depends_on": []},
    {"id": "PBI-02", "value": "null と空文字を0として集計し、行数は維持する。", "depends_on": ["PBI-01"]},  # 前提は、PBI-01
    {"id": "PBI-03", "value": "数値でない文字列を見つけたら、エラーを返す。", "depends_on": ["PBI-01"]},    # 前提は、PBI-01
]
# 1スプリント目の後に届く、利用者からの意見
FEEDBACK_AFTER_SPRINT1 = "空欄を含む売上が多いため、次は null と空文字への対応を優先してほしい。"

# 開発者が書くプログラムの決まり
CODE_CONTRACT = """
実行できるPythonスクリプト全体を返してください。
標準入力から1件のJSON {"amounts": [...]} を読み、標準出力にJSON {"total": 整数, "rows": 整数} だけを出力してください。
整数と整数の文字列、負の数を扱います。空の配列なら、合計・行数とも0です。
PBI-02 が対象なら、null と空文字を0に置き換え、行数は維持してください。
PBI-03 が対象なら、数値でない文字列に対して JSON {"error": "invalid_amount"} を返してください。
標準ライブラリの json と sys だけを使い、説明文やコードフェンスは付けないでください。
"""

# LLMのレスポンス形式を定義する（POが決めたバックログの順番）
class BacklogOrder(BaseModel):
    #  reason     ： この順番にした理由
    #  ordered_ids： 未完成の項目のIDを、価値の高い順に並べたもの
    reason: str = Field(description="この順番にした理由（日本語で1文）")
    ordered_ids: list[str]


#####################################
# 登場人物（PO 1人、SM 1人、開発者 3人）
#####################################
# 開発者どうしに上下は無く、3人で実装と品質に責任を持つ
PEOPLE = {
    "PO": "あなたはプロダクトオーナー（PO）です。価値とバックログの順番に責任を持ちます。",
    "SM": "あなたはスクラムマスター（SM）です。チームの進め方の改善を支援します。作業の割り当てはしません。",
    "開発者A": "あなたは開発者Aです。他の開発者と協力して、実装・レビュー・品質に共同で責任を持ちます。",
    "開発者B": "あなたは開発者Bです。他の開発者と協力して、実装・レビュー・品質に共同で責任を持ちます。",
    "開発者C": "あなたは開発者Cです。他の開発者と協力して、実装・レビュー・品質に共同で責任を持ちます。",
}

#####################################
# 登場人物に仕事を頼むチェーン（5人で共通）
#####################################
prompt_person = ChatPromptTemplate.from_messages([
    # system : アプリを作った開発者からの指示（その人物の役割と、今回の仕事）
    ("system", "{role}\n{instruction}"),
    # user   : チームで共有している情報
    ("user", "共有情報：\n{information}"),
])


# name の人物に仕事を頼む。output を渡すと、その形で答えを受け取る
def ask_person(name: str, instruction: str, information: dict, output=None):
    if output is None:
        chain_person = prompt_person | llm | StrOutputParser()
    else:
        chain_person = prompt_person | llm.with_structured_output(output, method="json_schema")
    return chain_person.invoke({
        "role": PEOPLE[name],
        "instruction": instruction,
        "information": json.dumps(information, ensure_ascii=False),
    })


#####################################
# 受け入れテスト：開発者のプログラムを実際に動かして確かめる
#####################################
# 注意：生成されたコードを、このPCで実際に実行します。
#      安全策は、学習用の簡易なもの（別のプロセスで実行する、APIキーなどの環境変数を渡さない、5秒で打ち切る）です。
def test_program(path: str, target_ids: list[str]) -> dict:
    tests = [
        {"name": "通常の集計", "input": ["10", "20", "5"], "expected": {"total": 35, "rows": 3}},
        {"name": "空データ", "input": [], "expected": {"total": 0, "rows": 0}},
    ]
    if "PBI-02" in target_ids:
        tests.append({"name": "空欄の補完", "input": ["10", None, "", "5"], "expected": {"total": 15, "rows": 4}})
    if "PBI-03" in target_ids:
        tests.append({"name": "不正値", "input": ["10", "?"], "expected": {"error": "invalid_amount"}})

    results = []
    for test in tests:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", path],
                input=json.dumps({"amounts": test["input"]}),
                env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
                capture_output=True, text=True, timeout=5,
            )
            actual = json.loads(completed.stdout) if completed.returncode == 0 else completed.stderr.strip()[-200:]
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            actual = str(error)
        results.append({"name": test["name"], "passed": actual == test["expected"], "actual": actual})
    return {"passed": all(result["passed"] for result in results), "tests": results}


# 生成されたコードを保存する（コードフェンスが付いていたら外す）
def save_code(code: str, filename: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        code = "\n".join(line for line in lines if not line.startswith("```"))
    os.makedirs(SCRUM_DIR, exist_ok=True)
    path = os.path.join(SCRUM_DIR, filename)
    with open(path, "w", encoding="utf-8") as file:
        file.write(code + "\n")
    return path


def main():
    """
    題材：
        PO1人・SM1人・開発者3人のスクラムチームで、2スプリント分の開発を短く再現する。

    登場人物：
        PO        ：プロダクトオーナー。利用者の意見を踏まえて、バックログ（作る機能の一覧）の順番を決める。
                    スプリントの終わりに、得られた価値と次の方向性を述べる。
        SM        ：スクラムマスター。スプリントの終わりに、チームの進め方の改善を提案する。作業の割り当てはしない。
        開発者A・B・C：3人で実装と品質に責任を持つ。スプリントごとに、次の3つの役割を交代で受け持つ。
                    実装担当  ：選ばれた項目を実装する。
                    レビュー担当：実装とテスト結果を読み、問題点と修正案を示す。
                    仕上げ担当  ：レビューを反映して、コードを仕上げる。

    前提：
        プロダクトの目標    ：売上データを、安全に集計できる小さなコマンドラインプログラムを作る。
        バックログ          ：PBI-01（正常な売上を集計する）、PBI-02（空欄を0として集計する）、PBI-03（不正値でエラーを返す）
        利用者の意見        ：1スプリント目の後に「次は空欄への対応を優先してほしい」という意見が届く。
        受け入れテスト      ：開発者のプログラムを、Pythonが実際に動かして確かめる。

    流れ（1スプリント分）：
        1. 計画             ：POがバックログの順番を決め、前提の項目が完成している項目から1つ選ぶ。
        2. 実装             ：実装担当が実装し、Pythonが受け入れテストを行う。
        3. レビュー         ：レビュー担当が、実装とテスト結果を読み、問題点と修正案を示す。
        4. 仕上げ           ：仕上げ担当がレビューを反映してコードを仕上げ、Pythonがもう一度テストする。合格すれば、項目は完成。
        5. スプリントレビュー：POが、得られた価値と次の方向性を述べる。
        6. 振り返り         ：SMが、次のスプリントで試す進め方の改善を提案する。
    """

    done_ids = []
    feedback = "まず、正常なデータの集計を使いたい。"
    accepted_code = "（なし）"
    developers = ["開発者A", "開発者B", "開発者C"]

    for sprint in (1, 2):
        print(f"========== スプリント{sprint} ==========")

        # 開発者3人の役割を、スプリントごとに交代する
        #   スプリント1：実装 A、レビュー B、仕上げ C
        #   スプリント2：実装 B、レビュー C、仕上げ A
        implementer = developers[(sprint - 1) % 3]
        reviewer = developers[sprint % 3]
        finisher = developers[(sprint + 1) % 3]
        print(f"[役割] 実装担当：{implementer}、レビュー担当：{reviewer}、仕上げ担当：{finisher}")

        # ##################################################################
        # 1. 計画：POがバックログの順番を決め、前提が完成済みの項目から1つ選ぶ
        # ##################################################################
        pending = [item for item in BACKLOG if item["id"] not in done_ids]
        result_order = ask_person(
            "PO",
            instruction="未完成の項目すべてのIDを、価値の高い順に並べてください。利用者の意見を重視してください。",
            information={"目標": PRODUCT_GOAL, "未完成の項目": pending, "利用者の意見": feedback},
            output=BacklogOrder,
        )
        print(f"[PO] 順番：{result_order.ordered_ids}（{result_order.reason}）")
        selected = next(item for item_id in result_order.ordered_ids for item in pending
                        if item["id"] == item_id and set(item["depends_on"]) <= set(done_ids))
        target_ids = done_ids + [selected["id"]]
        print(f"[計画] 今回の項目：{selected['id']}（{selected['value']}）")

        # ##################################################################
        # 2. 実装：実装担当が実装し、Pythonが受け入れテストを行う
        # ##################################################################
        code = ask_person(
            implementer,
            instruction=CODE_CONTRACT,
            information={"対象の項目": target_ids, "前回までのコード": accepted_code},
        )
        path = save_code(code, f"sprint{sprint}_v1.py")
        result_test = test_program(path, target_ids)
        print(f"[{implementer}] 実装：{path}")
        print(f"[テスト] {'合格' if result_test['passed'] else '不合格'}：{[(t['name'], t['passed']) for t in result_test['tests']]}")

        # ##################################################################
        # 3. レビュー：レビュー担当が、実装とテスト結果を読み、問題点と修正案を示す
        # ##################################################################
        review = ask_person(
            reviewer,
            instruction="コードとテスト結果を読み、問題点と修正案を、日本語で200字以内で示してください。問題が無ければ、良い点を述べてください。",
            information={"対象の項目": target_ids, "コード": code, "テスト結果": result_test},
        )
        print(f"[{reviewer}] レビュー：{review}")

        # ##################################################################
        # 4. 仕上げ：仕上げ担当がレビューを反映してコードを仕上げ、Pythonがもう一度テストする
        # ##################################################################
        code = ask_person(
            finisher,
            instruction=CODE_CONTRACT + "\nレビューを反映して仕上げた、スクリプト全体を返してください。",
            information={"対象の項目": target_ids, "現在のコード": code, "テスト結果": result_test, "レビュー": review},
        )
        path = save_code(code, f"sprint{sprint}_v2.py")
        result_test = test_program(path, target_ids)
        print(f"[{finisher}] 仕上げ：{path}")
        print(f"[テスト] {'合格' if result_test['passed'] else '不合格'}：{[(t['name'], t['passed']) for t in result_test['tests']]}")

        if result_test["passed"]:
            done_ids.append(selected["id"])
            accepted_code = code

        # ##################################################################
        # 5. スプリントレビュー：POが、得られた価値と次の方向性を述べる
        # ##################################################################
        feedback = FEEDBACK_AFTER_SPRINT1 if sprint == 1 else "追加の意見はありません。"
        po_review = ask_person(
            "PO",
            instruction="テスト結果と利用者の意見から、得られた価値と次の方向性を、日本語で150字以内で述べてください。",
            information={"完成した項目": done_ids, "テスト結果": result_test, "利用者の意見": feedback},
        )
        print(f"[PO] スプリントレビュー：{po_review}")

        # ##################################################################
        # 6. 振り返り：SMが、次のスプリントで試す進め方の改善を提案する
        # ##################################################################
        retro = ask_person(
            "SM",
            instruction="今回の実装・レビュー・仕上げの過程から、次のスプリントで試す進め方の改善を1つ、日本語で100字以内で提案してください。",
            information={"レビュー": review, "最終のテスト結果": result_test},
        )
        print(f"[SM] 振り返り：{retro}")

    print(f"完成した項目：{done_ids}")


if __name__ == "__main__":
    main()
