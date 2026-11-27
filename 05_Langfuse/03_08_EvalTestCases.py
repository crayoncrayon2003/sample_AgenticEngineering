import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel
from langfuse import Evaluation, get_client
from langfuse.langchain import CallbackHandler

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


llm = make_llm("low")

#####################################
# Langfuse：実行の記録（トレース）の送信先
#####################################
# ログの送信先を省略すると Langfuse Cloud（インターネット上）に送られる。ローカルの Langfuse に送信する
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()


#####################################
# テスト用の売上データ（架空）
#####################################
SALES_DATA = {
    "sales-A": [10, 20, 5],
    "sales-B": [-5, 10, 0],
    "sales-C": [],
}


###############################################################################
# ツール1 : 売上データを取得する
###############################################################################
@tool
def fetch_sales(dataset_id: str) -> list[int]:
    """指定した売上データを取得する。

    Args:
        dataset_id: 売上データのID。例：sales-A
    """
    return SALES_DATA[dataset_id]


###############################################################################
# ツール2 : 売上データの合計と行数を計算する
###############################################################################
@tool
def aggregate_sales(amounts: list[int]) -> dict:
    """売上データの合計と行数を計算する。空のデータは、合計も行数も0。

    Args:
        amounts: fetch_sales で取得した売上データ
    """
    return {"total": sum(amounts), "rows": len(amounts)}


#####################################
# LLMのレスポンス形式を定義する（最終回答）
#####################################
class SalesAnswer(BaseModel):
    #  dataset_id： 対象の売上データのID
    #  total     ： 合計
    #  rows      ： 行数
    dataset_id: str
    total: int
    rows: int


#####################################
# エージェント（評価の対象）
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[fetch_sales, aggregate_sales],
    # response_format: 最終回答の形（SalesAnswer）
    response_format=SalesAnswer,
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは売上を集計する係です。依頼されたデータを fetch_sales で取得し、"
        "取得した配列をそのまま aggregate_sales に渡してください。各ツールは1回ずつ使ってください。"
        "最終回答には、対象のIDと、ツールで得た合計・行数を入れてください。"
    ),
)


#####################################
# 評価：期待した結果と、実際の結果を比べる
#####################################
def evaluate(dataset_id: str, answer: dict, tool_names: list[str]) -> dict:
    amounts = SALES_DATA[dataset_id]
    expected = {"dataset_id": dataset_id, "total": sum(amounts), "rows": len(amounts)}
    checks = {
        "回答の数値とIDが正しい": answer == expected,
        "必要なツールを、正しい順番で1回ずつ使った": tool_names == ["fetch_sales", "aggregate_sales"],
    }
    return {"passed": all(checks.values()), "checks": checks}


#####################################
# Langfuse：テストケースをデータセットに登録し、実験として評価する
#####################################
DATASET_NAME = "03_08_sales_test_cases"


def register_dataset() -> None:
    langfuse.create_dataset(name=DATASET_NAME)
    for dataset_id in SALES_DATA:
        # id を決めておくと、何度実行しても同じテストケースが上書きされ、重複しない
        langfuse.create_dataset_item(dataset_name=DATASET_NAME, id=f"03_08-{dataset_id}", input={"dataset_id": dataset_id})


# 実験のタスク：テストケース1件分、エージェントを実行する
def task(*, item, **kwargs) -> dict:
    dataset_id = item.input["dataset_id"]
    result = agent.invoke({"messages": [{"role": "user", "content": f"{dataset_id} の売上合計と行数を調べてください。"}]},
                          config={"callbacks": [CallbackHandler()]})
    answer = result["structured_response"].model_dump()
    tool_names = [call["name"] for m in result["messages"] if m.type == "ai" for call in m.tool_calls]
    return {"answer": answer, "tool_names": tool_names}


# 実験の評価関数：evaluate() の確認結果を、スコア（1.0：合格、0.0：不合格）として返す
def score(*, input, output, **kwargs) -> list[Evaluation]:
    evaluation = evaluate(input["dataset_id"], output["answer"], output["tool_names"])
    return [Evaluation(name=name, value=1.0 if ok else 0.0) for name, ok in evaluation["checks"].items()]


def main():
    """
    題材：テストケースでエージェントを評価する。回答・ツールの使い方・時間・トークン数を記録し、Langfuse に保存する。
         テストケースは、合計3件ある（売上データ sales-A・sales-B・sales-C）。
         テストケースは Langfuse のデータセットに登録し、実験（run_experiment）として実行する。時間・トークン数はトレースに記録される。
         
         01_LangChain の 03_08_EvalTestCases.py
           テストケースは、コードの変数（ SALES_DATA ）に書く。
           for 文で1件ずつエージェントを実行して採点し、結果を JSON ファイルに保存する。

         05_Langfuse の 03_08_EvalTestCases.py
           1. 03_08_EvalTestCases.pyが、register_datasetを呼ぶ。
              register_datasetは、テストケースを Langfuse（サーバ） の データセットに登録する。
              登録するのは、テストケースの入力データだけである。テストケースのコードは送らない。
           2. 03_08_EvalTestCases.pyが、run_experiment を呼ぶ。run_experimentは、以下を実行する。
              - Langfuse（サーバ）のデータセットから、テストケースを取り出す。
              - 03_08_EvalTestCases.pyの中で、task（エージェントの実行）と score（採点）を呼ぶ。
              - 結果（トレースとスコア）を、Langfuse（サーバ）に送る。
           3. Langfuse（サーバ）は、テストケースと結果を保存する。テストの実行も採点もしない。
    """

    register_dataset()
    # max_concurrency=1：テストケースを1件ずつ順番に実行する
    result = langfuse.get_dataset(DATASET_NAME).run_experiment(
        name="売上集計エージェントの評価", task=task, evaluators=[score], max_concurrency=1
    )

    passed = 0
    for item_result in result.item_results:
        print("----------------------------------------")
        print(f"テスト：{item_result.item.input['dataset_id']}")
        checks = {e.name: e.value == 1.0 for e in item_result.evaluations}
        print(f"回答：{item_result.output['answer']}")
        print(f"使ったツール：{item_result.output['tool_names']}")
        print(f"評価：{'合格' if all(checks.values()) else '不合格'} {checks}")
        passed += all(checks.values())

    # 全体の結果をまとめて表示する。結果は Langfuse の実験の画面で確認できる
    print("----------------------------------------")
    print(f"合格：{passed} / {len(result.item_results)}件")
    print(f"保存先：{result.experiment_url}")

    # Langfuse：記録は裏でまとめて送られるため、プログラムを終える前に、残りの記録をすべて送る
    langfuse.flush()


if __name__ == "__main__":
    main()
