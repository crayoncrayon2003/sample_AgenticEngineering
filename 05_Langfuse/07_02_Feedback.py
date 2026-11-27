import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langfuse import get_client, propagate_attributes
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


#####################################
# Langfuse：実行の記録（トレース）の送信先
#####################################
# ログの送信先を省略すると Langfuse Cloud（インターネット上）に送られる。ローカルの Langfuse に送信する
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

llm = make_llm("minimal")


###############################################################################
# ツール1 : 指定した都市の今日の天気を返すツール。ただし、天気情報は仮想データ。
###############################################################################
@tool
def get_weather(city: str) -> str:
    """指定した都市の今日の天気を返す（模擬データ）。

    Args:
        city: 都市名。例：東京、大阪、札幌
    """
    weather = {
        "東京": "晴れ、最高気温18℃",
        "大阪": "くもり、最高気温17℃",
        "札幌": "雪、最高気温2℃",
    }
    return weather.get(city, f"{city}の天気データは登録されていません。")

###############################################################################
# ツール2 : 指定した都市への荷物の配送料金を計算するツール。ただし、料金情報は仮想データ。
###############################################################################
@tool
def get_shipping_fee(weight_kg: float, region: Literal["本州", "北海道", "沖縄"]) -> str:
    """荷物の重さと届け先の地域から、送料を計算する（模擬の料金表）。

    Args:
        weight_kg: 荷物の重さ（kg）
        region: 届け先の地域
    """
    base = {"本州": 800, "北海道": 1200, "沖縄": 1500}[region]
    extra = max(0, int(weight_kg) - 2) * 200
    return f"送料は{base + extra}円です（{region}、{weight_kg}kg）。"


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[get_weather, get_shipping_fee],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは案内係です。天気と送料は必ずツールで調べ、その結果だけを使って答えてください。"
        "ツールで分からないことは、分からないと伝えてください。"
        "これらは模擬データなので、実際の情報ではないことを添えてください。"
    ),
)


#####################################
# 自動の評価：回答が、決めたルールを守っているかを確かめる
#####################################
# 利用者の評価を待たずに、プログラムで確かめられることは、実行のたびに自動でスコアを付ける
def check_rule(answer: str) -> bool:
    """模擬データであることを、回答に添えているか。"""
    return "模擬" in answer


def main():
    """
    題材：天気と送料を調べられる案内エージェント（01_LangChain/02_01_ToolFunction.py と同じエージェント）の回答に、
         評価（スコア）を付けて Langfuse に送る。

    付けるスコア：
        1. 利用者の評価（user_feedback）：アプリの「役に立った・役に立たなかった」ボタンを押した、という想定。
           ここでは、押された結果をあらかじめ決めておく（実際のアプリでは、画面の操作で受け取る）。
        2. 自動の評価（mentions_mock_data）：回答に「模擬データ」であることを添えているかを、プログラムで確かめる。

    スコアの付け方：
        回答を作ったトレースのID（trace_id）を覚えておき、後から create_score でそのトレースにスコアを付ける。
        利用者の評価は、回答を返した後に届くため、このように「後から付ける」形になる。

    見るところ：
        画面のトレース一覧で、スコアの列が増える。「Scores」で、スコアごとの集計（役に立った割合など）を見られる。
        評価の低いトレースを絞り込んで、どんな質問で失敗したかを調べられる。
    """

    # 3件の質問と、利用者が押した評価（True：役に立った、False：役に立たなかった）とコメント
    conversations = [
        ("大阪の天気を教えてください。", True, "すぐに分かって便利でした。"),
        ("那覇の天気は？", False, "那覇の天気を知りたかったのに、分かりませんでした。"),
        ("3.5kgの荷物を北海道に送ると、送料はいくらですか？", True, ""),
    ]
    for question, helpful, comment in conversations:
        print("----------------------------------------")
        print(f"質問：{question}")

        # 1. 回答を作り、そのトレースのIDを覚えておく
        with langfuse.start_as_current_observation(name="案内エージェント", input=question) as span:
            with propagate_attributes(trace_name="案内エージェント", tags=["07_02_Feedback"]):
                result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config={"callbacks": [CallbackHandler()]})
            answer = result["messages"][-1].text
            span.update(output=answer)
            trace_id = langfuse.get_current_trace_id()
        print(f"回答：{answer}")

        # 2. 自動の評価を付ける
        rule_ok = check_rule(answer)
        langfuse.create_score(trace_id=trace_id, name="mentions_mock_data", value=1 if rule_ok else 0, data_type="BOOLEAN",
                              comment="模擬データであることを添えている" if rule_ok else "模擬データであることを添えていない")
        print(f"自動の評価：模擬データであることを{'添えている' if rule_ok else '添えていない'}")

        # 3. 利用者の評価を付ける（回答を返した後に届く想定）
        langfuse.create_score(trace_id=trace_id, name="user_feedback", value=1 if helpful else 0, data_type="BOOLEAN", comment=comment)
        print(f"利用者の評価：{'役に立った' if helpful else '役に立たなかった'}（{comment or 'コメントなし'}）")
        print(f"トレース：{langfuse.get_trace_url(trace_id=trace_id)}")

    # 送信は裏でまとめて行われる。プログラムを終える前に、残っている記録をすべて送る
    langfuse.flush()


if __name__ == "__main__":
    main()
