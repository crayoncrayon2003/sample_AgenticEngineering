import os

from dotenv import load_dotenv
from langchain.agents import create_agent

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


llm = make_llm("minimal")

#####################################
# エージェント1：調査担当
#####################################
agent_researcher = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール（今回は無し）
    tools=[],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたはAI分野の調査・整理担当です。テーマについて、主要な動向・実用例・影響・注意点を日本語で整理してください。"
        "検索ツールは無いため、あなたが持つ知識の範囲で答えてください。"
        "不確かな内容は不確かと明記し、確認していない出典やURLは作らないでください。"
    ),
)

#####################################
# エージェント2：執筆担当
#####################################
agent_writer = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール（今回は無し）
    tools=[],
    # system_prompt : エージェントに設定する役割
    system_prompt=(
        "あなたは技術記事の執筆担当です。渡された整理結果に基づいて、一般の読者向けのブログ記事を日本語で書いてください。"
        "タイトルと見出しを付け、200字程度にまとめてください。"
        "整理結果で不確かとされた内容は断定せず、根拠の無い数値や出典を追加しないでください。"
    ),
)


def main():
    """
    題材：2人のエージェントが順番に仕事を受け渡す。調査・整理担当が整理し、その結果を執筆担当が記事にする。

    登場人物：
      調査担当：テーマについて、主要な動向を整理する
      執筆担当：調査担当から受け取った結果を、ブログ記事にする
    """

    topic = "AIエージェントの主要な動向"
    print(f"テーマ：{topic}")

    # 1段目：調査・整理担当が、テーマを整理する
    result_step1 = agent_researcher.invoke({"messages": [{"role": "user", "content": (
        f"テーマ：{topic}\n主要な動向を3つ選び、それぞれ要点・実用例・影響・注意点を整理してください。"
    )}]})
    findings = result_step1["messages"][-1].text
    print("--- 1段目：調査・整理担当の整理結果 ---")
    print(findings)

    # 2段目：執筆担当が、整理結果を受け取って記事にする
    result_step2 = agent_writer.invoke({"messages": [{"role": "user", "content": (
        f"テーマ：{topic}\n次の整理結果に基づいて、ブログ記事を書いてください。\n\n【整理結果】\n{findings}"
    )}]})
    print("--- 2段目：執筆担当の記事 ---")
    print(result_step2["messages"][-1].text)


if __name__ == "__main__":
    main()
