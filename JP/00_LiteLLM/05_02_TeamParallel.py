import json
import os
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


llm = make_llm("low")  # minimalだと、ツールを使わずに答えることがある

#####################################
# 架空の調査データ
#####################################
# 分野ごとのニュース（架空）
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
# 分野ごとの統計（架空）。炭素回収の統計は無い
STATISTICS = {
    "再生可能エネルギー": "発電量に占める再生可能エネルギーの割合：2020年 20% → 2025年 28%。太陽光パネルの設置費用：5年で3割減。",
    "電気自動車": "新車販売に占める電気自動車の割合：2020年 4% → 2025年 15%。",
}


#####################################
# ツール1 : 分野ごとのニュースを検索する
#####################################
def search_news(field: str) -> str:
    """指定した分野の最近のニュースを検索する（架空の模擬データ）。

    Args:
        field: 分野。「再生可能エネルギー」「電気自動車」「炭素回収」のいずれか
    """
    news = NEWS.get(field)
    if not news:
        return f"{field}のニュースは見つかりませんでした。"
    return "\n".join(f"- {item}" for item in news)


#####################################
# ツール2 : 分野ごとの統計データを取得する
#####################################
def get_statistics(field: str) -> str:
    """指定した分野の統計データを取得する（架空の模擬データ）。

    Args:
        field: 分野。「再生可能エネルギー」「電気自動車」「炭素回収」のいずれか
    """
    return STATISTICS.get(field, f"{field}の統計データはありません。")


#####################################
# LLMに渡すツールの説明
#####################################
# LangChain の @tool は、関数の docstring と型ヒントからこの説明を自動で作る。ここでは辞書で書く
tools = [
    {"type": "function", "function": {
        "name": "search_news",
        "description": "指定した分野の最近のニュースを検索する（架空の模擬データ）。",
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string", "description": "分野。「再生可能エネルギー」「電気自動車」「炭素回収」のいずれか"}},
            "required": ["field"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_statistics",
        "description": "指定した分野の統計データを取得する（架空の模擬データ）。",
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string", "description": "分野。「再生可能エネルギー」「電気自動車」「炭素回収」のいずれか"}},
            "required": ["field"],
        },
    }},
]
tool_functions = {"search_news": search_news, "get_statistics": get_statistics}


#####################################
# 調査担当（分野ごとに1人）
#####################################
# 調査担当の system_prompt を作る（どのツールを何回使うかは、エージェントが決める）
def make_system_prompt(field: str) -> str:
    return (
        f"あなたは「{field}」の調査担当です。\n"
        "- search_news でニュースを調べ、get_statistics で統計データを調べてください。\n"
        "- ニュースの数値と統計データが食い違う場合は、統計データを優先し、食い違いがあることを書いてください。\n"
        "- 統計データが無い場合は、統計で確かめられなかったことを書いてください。\n"
        "調べた結果を、日本語の2〜3文で要約してください。架空の模擬データであることも添えてください。"
    )


# 調査担当に依頼し、回答と、使ったツールの一覧を返す（LLM → ツール実行 → LLM のループ）
def ask(field: str, request: str) -> dict:
    messages = [
        {"role": "system", "content": make_system_prompt(field)},
        {"role": "user", "content": request},
    ]
    tool_calls = []
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_calls.append(tool_call.function.name)
            result = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    return {"answer": response.content, "tool_calls": tool_calls}



#####################################
# 1段目：3人の調査担当が、同時に調べる
#####################################
# ThreadPoolExecutor：処理を別々のスレッドで同時に動かす（LangChain の RunnableParallel に当たる）
def chain_parallel(request: str) -> dict:
    with ThreadPoolExecutor() as executor:
        future_energy = executor.submit(ask, "再生可能エネルギー", request)
        future_ev = executor.submit(ask, "電気自動車", request)
        future_carbon = executor.submit(ask, "炭素回収", request)
        return {"energy": future_energy.result(), "ev": future_ev.result(), "carbon": future_carbon.result()}


#####################################
# 2段目：3人の調査結果をまとめる
#####################################
prompt_merge = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "あなたは調査結果をまとめる担当です。渡された3つの調査結果だけを使い、日本語の短いレポートを作ってください。"
        "自分の知識で情報を追加せず、どの分野の結果かを明確にしてください。架空のデータであることも添えてください。"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": (
        "【再生可能エネルギー】\n{energy}\n\n【電気自動車】\n{ev}\n\n【炭素回収】\n{carbon}"
    )},
]

# LangChain の chain_merge.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_merge(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_merge, values))
    return message.content


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

    # 1段目：3人の調査担当が、同時に調べる
    result_step1 = chain_parallel(request)
    print("--- 1段目：3人の調査結果 ---")
    print(f"[再生可能エネルギー] 使ったツール：{result_step1['energy']['tool_calls']}")
    print(result_step1["energy"]["answer"])
    print(f"[電気自動車] 使ったツール：{result_step1['ev']['tool_calls']}")
    print(result_step1["ev"]["answer"])
    print(f"[炭素回収] 使ったツール：{result_step1['carbon']['tool_calls']}")
    print(result_step1["carbon"]["answer"])

    # 2段目：まとめ担当が、1つのレポートにする
    result_step2 = chain_merge({
        "energy": result_step1["energy"]["answer"],
        "ev": result_step1["ev"]["answer"],
        "carbon": result_step1["carbon"]["answer"],
    })
    print("--- 2段目：まとめたレポート ---")
    print(result_step2)


if __name__ == "__main__":
    main()
