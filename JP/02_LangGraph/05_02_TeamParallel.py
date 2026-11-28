import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task

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


llm = make_llm("low")  # minimalだと、ツールを使わずに答えることがある


###############################################################################
# 架空の調査データ：分野ごとのニュースデータ
###############################################################################
NEWS = {
    "再生可能エネルギー": [
        "ある地域で洋上風力発電所の運転が始まり、家庭10万世帯分の電力をまかなう見込み。",
        "太陽光パネルの設置費用が、5年前より4割下がったという報道があった。",
    ],
    "電気自動車": [
        "ある企業が、10分で8割まで充電できる電池の試作に成功した。量産は未定。",
    ],
    "炭素回収": [
        "空気中の二酸化炭素を回収する設備の費用が、試算で3割下がった。実証は小規模にとどまる。",
    ],
}
###############################################################################
# 架空の調査データ：分野ごとの統計データ。ただし、炭素回収は無い状況を想定する
###############################################################################
STATISTICS = {
    "再生可能エネルギー": "発電量に占める再生可能エネルギーの割合：2020年 20% → 2025年 28%。太陽光パネルの設置費用：5年で3割減。",
    "電気自動車": "新車販売に占める電気自動車の割合：2020年 4% → 2025年 15%。",
}


###############################################################################
# ツール1 : 分野ごとのニュースを検索する
###############################################################################
@tool
def search_news(field: str) -> str:
    """指定した分野の最近のニュースを検索する（架空の模擬データ）。

    Args:
        field: 分野。「再生可能エネルギー」「電気自動車」「炭素回収」のいずれか
    """
    news = NEWS.get(field)
    if not news:
        return f"{field}のニュースは見つかりませんでした。"
    return "\n".join(f"- {item}" for item in news)


###############################################################################
# ツール2 : 分野ごとの統計データを取得する
###############################################################################
@tool
def get_statistics(field: str) -> str:
    """指定した分野の統計データを取得する（架空の模擬データ）。

    Args:
        field: 分野。「再生可能エネルギー」「電気自動車」「炭素回収」のいずれか
    """
    return STATISTICS.get(field, f"{field}の統計データはありません。")


#####################################
# 調査担当のエージェントを作る（分野ごとに1人）
#####################################
def make_agent_researcher(field: str):
    return create_agent(
        # model　　　　　: エージェントに設定するLLM
        model=llm,
        # tools         : エージェントに設定するツール（どのツールを何回使うかは、エージェントが決める）
        tools=[search_news, get_statistics],
        # system_prompt : エージェントに設定する役割
        system_prompt=(
            f"あなたは「{field}」の調査担当です。\n"
            "- search_news でニュースを調べ、get_statistics で統計データを調べてください。\n"
            "- ニュースの数値と統計データが食い違う場合は、統計データを優先し、食い違いがあることを書いてください。\n"
            "- 統計データが無い場合は、統計で確かめられなかったことを書いてください。\n"
            "調べた結果を、日本語の2〜3文で要約してください。架空の模擬データであることも添えてください。"
        ),
    )


agent_energy = make_agent_researcher("再生可能エネルギー")
agent_ev = make_agent_researcher("電気自動車")
agent_carbon = make_agent_researcher("炭素回収")


# エージェントに依頼する。エージェントは、回答と使ったツールの一覧を返す
def ask(agent, request: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    tool_calls = [call["name"] for message in result["messages"] if message.type == "ai" for call in message.tool_calls]
    return {"answer": result["messages"][-1].text, "tool_calls": tool_calls}


# 調査担当ごとの依頼（RunnableParallel に渡す例なので、1人ずつ関数にする）
def research_energy(request: str) -> dict:
    return ask(agent_energy, request)


def research_ev(request: str) -> dict:
    return ask(agent_ev, request)


def research_carbon(request: str) -> dict:
    return ask(agent_carbon, request)

#####################################
# 2段目：3人の調査結果をまとめる
#####################################
prompt_merge = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは調査結果をまとめる担当です。渡された3つの調査結果だけを使い、日本語の短いレポートを作ってください。"
        "自分の知識で情報を追加せず、どの分野の結果かを明確にしてください。架空のデータであることも添えてください。"
    ),(
        # user   : 3人の調査結果
        "user",
        "【再生可能エネルギー】\n{energy}\n\n【電気自動車】\n{ev}\n\n【炭素回収】\n{carbon}"
    ),
])
chain_merge = prompt_merge | llm | StrOutputParser()



#####################################
# タスク（ワークフローの1ステップ）
#####################################
@task
def task_energy(request: str) -> dict:
    return research_energy(request)


@task
def task_ev(request: str) -> dict:
    return research_ev(request)


@task
def task_carbon(request: str) -> dict:
    return research_carbon(request)


@task
def merge(energy: str, ev: str, carbon: str) -> str:
    return chain_merge.invoke({"energy": energy, "ev": ev, "carbon": carbon})


#####################################
# ワークフロー：3人が同時に調べる → まとめる
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # 1段目：3つのタスクを開始する。.result() を呼ぶまで待たないので、3人が同時に調べる
    future_energy = task_energy(request)
    future_ev = task_ev(request)
    future_carbon = task_carbon(request)
    energy = future_energy.result()
    ev = future_ev.result()
    carbon = future_carbon.result()

    # 2段目：3人の調査結果をまとめる
    report = merge(energy["answer"], ev["answer"], carbon["answer"]).result()
    return {"energy": energy, "ev": ev, "carbon": carbon, "report": report}


def main():
    """
    題材：
        分野の違う3人の調査担当が、同時にそれぞれの分野を調べる。
        まとめ担当が、3人の調査結果を1つのレポートにまとめる。

    登場人物：
        調査担当（3人）：再生可能エネルギー・電気自動車・炭素回収を、1人1分野ずつ担当する。
                       ニュース検索と統計データの2つのツールを持ち、どれを何回使うかを自分で決める。
        まとめ担当     ：3人の調査結果だけを使って、1つのレポートにまとめる。

    前提：
        データ：ニュース検索と統計データは、どちらも架空の模擬データ。
               ニュース検索と統計データは、次の相違点がある
                - 再生可能エネルギーは、ニュース検索で設置費用は4割下がっている/統計データで設置費用は3割下下がっている
                - 炭素回収は、ニュース検索のデータがある/統計データのデータがない。
        そのため、同じ指示を受けても、調査担当ごとに調べ方と報告の内容が変わる。

    流れ：
        1. 3人の調査担当が、同時に調べる。各担当は、必要なツールを自分で選んで使い、2〜3文で要約する。
        2. 各担当が使ったツールと、調査結果を表示する。
        3. まとめ担当が、3人の調査結果を1つのレポートにまとめる。
    """

    request = "持続可能な技術の、最近の進展を調べてください。"
    print(f"依頼：{request}")

    # 1段目（3人が同時に調べる）と2段目（まとめる）を、まとめて実行する
    result = workflow.invoke(request)
    print("--- 1段目：3人の調査結果 ---")
    print(f"[再生可能エネルギー] 使ったツール：{result['energy']['tool_calls']}")
    print(result["energy"]["answer"])
    print(f"[電気自動車] 使ったツール：{result['ev']['tool_calls']}")
    print(result["ev"]["answer"])
    print(f"[炭素回収] 使ったツール：{result['carbon']['tool_calls']}")
    print(result["carbon"]["answer"])
    print("--- 2段目：まとめたレポート ---")
    print(result["report"])


if __name__ == "__main__":
    main()
