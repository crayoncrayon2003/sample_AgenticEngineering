import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
TICKET_DIR = os.path.join(ROOT, "outputs", "tickets")


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


###############################################################################
# ツール1 : 基本的な確認手順を返す。人間の承認は不要
###############################################################################
@tool
def troubleshoot_issue(issue: str) -> str:
    """問題に対する、基本的な確認手順を返す。

    Args:
        issue: 問題の内容
    """
    return "確認手順：1. 接続状態を確認する。2. 表示されたエラーメッセージを記録する。"


###############################################################################
# ツール2 : 問い合わせ票を作る。人間の承認は必要
###############################################################################
@tool
def create_ticket(issue_type: str, details: str) -> str:
    """問い合わせ票を作成し、ファイルに保存する。

    Args:
        issue_type: 問題の種類
        details: 問題の詳しい内容
    """
    os.makedirs(TICKET_DIR, exist_ok=True)
    ticket_id = f"TICKET-{uuid4().hex[:8]}"
    with open(os.path.join(TICKET_DIR, f"{ticket_id}.json"), "w", encoding="utf-8") as file:
        json.dump({"ticket_id": ticket_id, "issue_type": issue_type, "details": details}, file, ensure_ascii=False, indent=2)
    return f"問い合わせ票 {ticket_id} を作成しました。"


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[troubleshoot_issue, create_ticket],
    # middleware    : create_ticket を実行する前に止めて、人間の判断を待つ
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"create_ticket": True})],
    # checkpointer  : 途中で止めた状態を保存する（再開するときに必要）
    checkpointer=InMemorySaver(),
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは電子機器の技術サポート係です。"
        "解決しない問題を記録するよう頼まれたら、create_ticketで問い合わせ票を作成してください。"
        "作成が却下された場合は、作成しなかったことを伝えてください。日本語で簡潔に答えてください。"
    ),
)


def main():
    """
    題材：AIが問い合わせ票を作る前に、処理を止めて人間の担当者に承認を求める。
         承認されたら作成し、却下されたら作成しない。

    登場人物：
      AIアシスタント：このプログラムのエージェント。技術サポートを行い、問い合わせ票を作るツールを持つ
      田中さん      ：田中さんは、AIアシスタントへ、プリンターの不具合を相談する
      担当者        ：人間のサポート担当者。AIアシスタントが問い合わせ票を作る前に、承認または却下する
    """

    # thread_id：止めた処理を再開するときに、どの会話かを示すID
    config = {"configurable": {"thread_id": "support-1"}}

    # ##################################################################
    # 田中さんの相談：AIアシスタントは問い合わせ票を作ろうとして、止まる
    # ##################################################################
    message = "プリンターの認証エラーが再起動しても直りません。問い合わせ票を作って記録してください。"
    print(f"[田中さん] {message}")
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)

    # __interrupt__ があれば、人間の判断を待って止まっている
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        for action in request["action_requests"]:
            print(f"[承認待ち] ツール：{action['name']}")
            print(f"[承認待ち] 引数　：{json.dumps(action['args'], ensure_ascii=False)}")

        # ##################################################################
        # 担当者の判断：承認（y）か却下（n）を入力する
        # ##################################################################
        answer = input("[担当者] 問い合わせ票の作成を承認しますか？（y/n）：").strip().lower()
        if answer == "y":
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject", "message": "担当者が作成を却下しました。"}

        # Command(resume=...) で、担当者の判断を渡して処理を再開する
        result = agent.invoke(Command(resume={"decisions": [decision]}), config)

    print(f"[AIアシスタント] {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
