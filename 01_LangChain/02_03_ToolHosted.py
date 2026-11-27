import os

from dotenv import load_dotenv
from langchain.agents import create_agent

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")


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
            timeout=180,
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
            timeout=180,
            max_retries=1,
        )
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


llm = make_llm("low")

#####################################
# LLM提供元（OpenAI／Anthropic）が提供する組み込みツール（提供元ごとに書き方が違う）
#####################################
# 組込みツールは、辞書型で指定する。提供元のサーバーでツールが動く
if PROVIDER == "openai":
    tool_web_search = {"type": "web_search"}
    tool_code_execution = {"type": "code_interpreter", "container": {"type": "auto"}}
    tool_file_search = {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}
else:
    tool_web_search = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
    tool_code_execution = {"type": "code_execution_20260521", "name": "code_execution"}
    tool_file_search = None

#####################################
# エージェント
#####################################
# Web検索エージェント、ツール tool_web_search を利用する
agent_web = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[tool_web_search],
    # system_prompt : エージェントに設定する役割
    system_prompt="必ずWeb検索を使い、一次資料を優先して、日付と出典付きで答えてください。",
)

# コード実行エージェント、ツール tool_code_execution を利用する
agent_code = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[tool_code_execution],
    # system_prompt : エージェントに設定する役割
    system_prompt="計算は必ずコード実行ツールでPythonを実行して行い、結果を簡潔に答えてください。",
)
# ファイル検索行エージェント、OpenAI 専用、ツール tool_code_execution を利用する
if PROVIDER == "openai" and VECTOR_STORE_ID:
    agent_file = create_agent(
        model=llm,
        tools=[tool_file_search],
        system_prompt="必ずfile_searchで調べ、見つからない場合は『検索結果では確認できませんでした』と答えてください。",
    )
else:
    agent_file = None


#####################################
# 質問して、回答の中身を表示する
#####################################
# エージェントに質問し、回答を表示して返す
def ask(agent, question: str):
    print(f"質問：{question}")
    # エージェントを実行する（"user"：アプリを使う利用者の発言）
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1]
    print(f"回答：{answer.text}")
    return answer


#####################################
# 結果を表示する
#####################################
# Web検索エージェント／ファイル検索エージェントの応答を表示する
def show_citations(answer) -> None:
    for block in answer.content_blocks:
        for citation in block.get("annotations", []):
            print(f"  出典：{citation.get('title', '')} {citation.get('url', '')}")


# コード実行エージェントの応答を表示する
def show_executed_code(answer) -> None:
    for block in answer.content_blocks:
        if block.get("type") == "server_tool_call":
            print("  [実行されたコード]")
            print(block.get("args", {}).get("code", ""))


def main():
    """
    題材：提供元の組み込みツールを使う3つのエージェント。
         1. Web検索      ：出典付きで最近の話題を答える
         2. コード実行   ：提供元のサーバーでPythonを実行して計算する
         3. ファイル検索 ：OpenAIに登録した文書を検索して答える（OpenAI専用）
    """

    print("========== 1. Web検索 ==========")
    answer = ask(agent_web, "Pythonの最新の安定版のバージョンと、そのリリース日を教えてください。")
    show_citations(answer)

    print("========== 2. コード実行 ==========")
    answer = ask(agent_code, "30の階乗の各桁の数字を合計してください。")
    show_executed_code(answer)

    print("========== 3. ファイル検索 ==========")
    if agent_file is None:
        print("OpenAI 以外、または .env に OPENAI_VECTOR_STORE_ID が無いため、実行できません。")
        return

    for question in ["第2四半期の売上目標は？", "研究室Xの電話番号は？"]:
        answer = ask(agent_file, question)
        show_citations(answer)


if __name__ == "__main__":
    main()
