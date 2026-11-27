import json
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task
from langgraph.types import interrupt

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
# 1段目：相談から、問い合わせ票の下書きを作るチェーン
#####################################
# LLMのレスポンス形式を定義する（問い合わせ票の下書き）
class TicketDraft(BaseModel):
    #  issue_type： 問題の種類
    #  details   ： 問題の詳しい内容
    issue_type: str = Field(description="問題の種類。例：プリンターの認証エラー")
    details: str = Field(description="問題の詳しい内容と、試したこと")


prompt_draft = ChatPromptTemplate.from_messages([
    # system : アプリを作った開発者からの指示
    ("system", "あなたは電子機器の技術サポート係です。利用者の相談から、問い合わせ票の下書きを作ってください。"),
    # user   : アプリを使う利用者の発言
    ("user", "{message}"),
])
chain_draft = prompt_draft | llm.with_structured_output(TicketDraft, method="json_schema")

#####################################
# 最後の段：利用者への返事を作るチェーン
#####################################
prompt_reply = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "あなたは電子機器の技術サポート係です。結果に基づいて、利用者への返事を日本語で簡潔に作ってください。"
        "作成が却下された場合は、作成しなかったことを伝えてください。"
    ),(
        # user   : 相談と、問い合わせ票の結果
        "user",
        "【相談】\n{message}\n\n【問い合わせ票の結果】\n{ticket_result}"
    ),
])
chain_reply = prompt_reply | llm | StrOutputParser()


#####################################
# タスク（ワークフローの1ステップ）
#####################################
# 途中で止めたときに結果が保存されるため、Pydantic のオブジェクトではなく、辞書で返す
@task
def draft_ticket(message: str) -> dict:
    return chain_draft.invoke({"message": message}).model_dump()


@task
def reply(message: str, ticket_result: str) -> str:
    return chain_reply.invoke({"message": message, "ticket_result": ticket_result})


#####################################
# ワークフロー：下書き → 承認を待つ（interrupt）→ 作成 → 返事
#####################################
# checkpointer：止めた状態を保存する（再開するときに必要）
@entrypoint(checkpointer=InMemorySaver())
def workflow(message: str) -> str:
    # 1段目：問い合わせ票の下書きを作る
    draft = draft_ticket(message).result()

    # 2段目：処理を止めて、担当者の判断を待つ。再開したときに渡された値（y/n）が、ここに返る
    decision = interrupt({"issue_type": draft["issue_type"], "details": draft["details"]})

    # 3段目：承認されたときだけ、問い合わせ票を作る
    if decision == "y":
        ticket_result = create_ticket.invoke({"issue_type": draft["issue_type"], "details": draft["details"]})
    else:
        ticket_result = "担当者が作成を却下しました。"

    # 最後の段：利用者への返事を作る
    return reply(message, ticket_result).result()


def main():
    """
    題材：AIが問い合わせ票を作る前に、処理を止めて人間の担当者に承認を求める。
         承認されたら作成し、却下されたら作成しない。

    登場人物：
      AIアシスタント：このプログラムのワークフロー。問い合わせ票の下書きを作り、承認されたら作成する
      田中さん      ：田中さんは、AIアシスタントへ、プリンターの不具合を相談する
      担当者        ：人間のサポート担当者。AIアシスタントが問い合わせ票を作る前に、承認または却下する
    """

    # thread_id：止めた処理を再開するときに、どの会話かを示すID
    config = {"configurable": {"thread_id": "support-1"}}

    # ##################################################################
    # 田中さんの相談：AIアシスタントは問い合わせ票の下書きを作り、承認を待って止まる
    # ##################################################################
    message = "プリンターの認証エラーが再起動しても直りません。問い合わせ票を作って記録してください。"
    print(f"[田中さん] {message}")
    result = workflow.invoke(message, config)

    # __interrupt__ があれば、人間の判断を待って止まっている
    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"[承認待ち] 問題の種類：{request['issue_type']}")
        print(f"[承認待ち] 詳しい内容：{request['details']}")

        # ##################################################################
        # 担当者の判断：承認（y）か却下（n）を入力する
        # ##################################################################
        answer = input("[担当者] 問い合わせ票の作成を承認しますか？（y/n）：").strip().lower()

        # Command(resume=...) で、担当者の判断を渡して処理を再開する。渡した値が interrupt() の戻り値になる
        result = workflow.invoke(Command(resume=answer), config)

    print(f"[AIアシスタント] {result}")


if __name__ == "__main__":
    main()
