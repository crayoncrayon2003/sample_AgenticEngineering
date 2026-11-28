import os
import sys
from dataclasses import dataclass
from uuid import uuid4

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic

# 埋め込みベクトルは OpenAI にしかない機能のため、Claude では実行できない
if PROVIDER != "openai":
    sys.exit("このサンプルは埋め込みベクトル（OpenAI専用）を使うため、LLM_PROVIDER=openai で実行してください。")


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
# 長期記憶の保存場所
#####################################
# "text" 項目を埋め込みベクトルにして、意味で検索できるようにする
store = InMemoryStore(index={
    "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
    "dims": 1536,
    "fields": ["text"],
})


#####################################
# 実行時に渡す情報：誰との会話か
#####################################
@dataclass
class UserContext:
    user_id: str


#####################################
# 記憶を抜き出すチェーン
#####################################
# LLMのレスポンス形式を定義する（抜き出した事実）
class NewFacts(BaseModel):
    #  facts： 利用者の発言から抜き出した、利用者自身についての事実の一覧
    facts: list[str] = Field(
        description="利用者の名前・好み・予定など、今後の会話でも役立つ情報。"
        "1件1文で「利用者は〜」の形にする。質問だけで新しい情報が無ければ空のリスト"
    )


prompt_extract = ChatPromptTemplate.from_messages([
    (
        # system : アプリを作った開発者からの指示
        "system",
        "利用者の発言から、利用者自身についての新しい事実だけを抜き出してください。\n"
        "- 質問や依頼だけの発言からは、何も抜き出さない（空のリスト）。\n"
        "- 既存の記憶と同じ内容は、抜き出さない。\n"
        "例：\n"
        "  「佐藤です。猫を飼っています。」→ [\"利用者の名前は佐藤。\", \"利用者は猫を飼っている。\"]\n"
        "  「私の名前を覚えていますか？」→ []",
    ),
    # user   : アプリを使う利用者の発言
    ("user", "【既存の記憶】{known}\n【利用者の発言】{message}"),
])
# 抜き出しは判断が難しいため、推論を少し深くしたモデルを使う
chain_extract = prompt_extract | make_llm("low").with_structured_output(NewFacts, method="json_schema")


# 記憶を入れたシステムプロンプトを作る
def make_system_prompt(memories: str) -> str:
    return (
        "あなたは利用者専属のアシスタントです。日本語で簡潔に答えてください。\n"
        "利用者について分かっていることは、次の記憶だけです。"
        "記憶に無いことは、知らないと答えてください。\n"
        f"【記憶】\n{memories}"
    )


#####################################
# タスク（ワークフローの1ステップ）
#####################################
# LLMを呼ぶ
@task
def call_model(system_prompt: str, messages: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=system_prompt)] + messages)


# 利用者の発言から、記憶すべき事実を抜き出す
@task
def extract_facts(known: list[str], message: str) -> NewFacts:
    return chain_extract.invoke({"known": known, "message": message})


#####################################
# エージェント：思い出す → 回答する → 記憶する
#####################################
# store：会話をまたぐ長期記憶の保存先、context_schema：実行時に渡す情報の形（誰との会話か）
@entrypoint(checkpointer=InMemorySaver(), store=store, context_schema=UserContext)
def agent(message: str, *, previous: list | None = None, runtime: Runtime[UserContext]) -> dict:
    # 記憶を利用者ごとに分けるための名前空間
    namespace = ("users", runtime.context.user_id)
    history = add_messages(previous or [], [HumanMessage(content=message)])

    # Step1 思い出す：発言に近い記憶を検索する
    items = runtime.store.search(namespace, query=message, limit=3)
    memories = "\n".join(f"- {item.value['text']}" for item in items) or "（なし）"
    print(f"  [思い出した記憶] {[item.value['text'] for item in items]}")

    # Step2 回答する
    response = call_model(make_system_prompt(memories), history).result()
    history = add_messages(history, [response])

    # Step3 記憶する：発言から事実を抜き出して保存する
    known = [item.value["text"] for item in runtime.store.search(namespace, limit=20)]
    result_extract = extract_facts(known, message).result()
    for fact in result_extract.facts:
        runtime.store.put(namespace, uuid4().hex, {"text": fact})
        print(f"  [記憶した] {fact}")

    # value：今回の戻り値、save：次回の previous として保存する会話履歴
    return entrypoint.final(value={"answer": response.text}, save=history)


def main():
    """
    題材：AIアシスタントが、利用者から聞いた名前・好み・予定を記憶し、別の会話で思い出して使う。

    登場人物：
      AIアシスタント             ：このプログラムのエージェント。田中さん・鈴木さんとの会話を区別して記憶する
      田中さん（利用者ID：tanaka）：田中さんは、AIアシスタントへ、自分の名前・説明の好み・旅行の予定を伝える
      鈴木さん（利用者ID：suzuki）：鈴木さんは、AIアシスタントへ、自分の名前・説明の好み・旅行の予定を伝えない
    """

    # 5件の発言（利用者のID、会話のID、発言）
    #  会話1（thread-1）：田中さんは、AIアシスタントへ、自分の名前・説明の好み・コード例の言語・旅行の予定を伝える。
    #                    AIアシスタントは、田中さんの記憶として保存する。
    #  会話2（thread-2）：田中さんは、会話1とは異なる会話2で、AIアシスタントに「私の好みや予定を覚えているか」を尋ねる。
    #                    AIアシスタントは、田中さんの記憶を検索し、会話1で聞いた内容を答える。
    #  会話3（thread-3）：鈴木さんは、AIアシスタントに「私の旅行の予定は？」と尋ねる。
    #                    AIアシスタントは、鈴木さんの記憶だけを検索する。田中さんの情報を記憶していないため、知らないと答える。
    messages = [
        ("tanaka", "thread-1", "田中です。説明は結論から短めが好みです。コード例は、Pythonで提示してください。"),
        ("tanaka", "thread-1", "来月、京都に旅行する予定です。"),
        ("tanaka", "thread-2", "私に合った説明の仕方とコード例の言語を覚えていますか？"),  # "thread-id"が異なる
        ("tanaka", "thread-2", "私の旅行の予定は？"),                                   # "thread-id"が異なる
        ("suzuki", "thread-3", "私の旅行の予定は？"),
    ]
    for user_id, thread_id, message in messages:
        print("----------------------------------------")
        print(f"[{user_id} / {thread_id}] 利用者：{message}")
        # thread_id：どの会話の続きかを表すID
        config = {"configurable": {"thread_id": thread_id}}

        # エージェントを実行する
        result = agent.invoke(message, config, context=UserContext(user_id=user_id))
        answer = result["answer"]
        print(f"[{user_id} / {thread_id}] AI：{answer}")

    # 田中さんの記憶を、意味で直接検索する（「説明の好み」に近いものから並ぶ）
    print("--- 田中さんの記憶を「説明の好み」で検索 ---")
    for item in store.search(("users", "tanaka"), query="説明の好み", limit=5):
        print(f"  類似度 {item.score:.2f}：{item.value['text']}")


if __name__ == "__main__":
    main()
