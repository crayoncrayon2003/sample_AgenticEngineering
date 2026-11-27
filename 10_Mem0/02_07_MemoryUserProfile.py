import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, after_agent, dynamic_prompt
from langgraph.checkpoint.memory import InMemorySaver
# mem0 の利用状況の送信（テレメトリ）を止め、mem0 が作る設定フォルダ（既定は ~/.mem0）を、このフォルダの outputs に置く。
# どちらも mem0 を import する前に設定する
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("MEM0_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "mem0"))
from mem0 import Memory
from qdrant_client import QdrantClient

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
# 長期記憶の保存場所（mem0）
#####################################
# mem0 は、発言から事実を抜き出し、埋め込みベクトルにして保存し、意味で検索する仕組みをまとめて引き受ける。
#   llm          ：事実を抜き出すLLM。LangChain のモデルをそのまま渡せる（抜き出しは判断が難しいため、推論を少し深くする）
#   embedder     ：文章を埋め込みベクトルにする方法
#   vector_store ：ベクトルの保存先。ここでは Qdrant を、プログラムの実行中だけメモリに置く形で使う（InMemoryStore と同じ）
#   history_db_path：記憶の変更履歴（追加・更新・削除）の保存先
#   custom_instructions：事実の抜き出し方の指示
os.makedirs(os.environ["MEM0_DIR"], exist_ok=True)
memory = Memory.from_config({
    "llm": {"provider": "langchain", "config": {"model": make_llm("low")}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {"provider": "qdrant", "config": {"client": QdrantClient(location=":memory:"), "embedding_model_dims": 1536}},
    "history_db_path": os.path.join(os.environ["MEM0_DIR"], "history.db"),
    "custom_instructions": (
        "利用者の名前・好み・予定など、今後の会話でも役立つ、利用者自身についての事実だけを抜き出す。"
        "事実は日本語で、1件1文の「利用者は〜」の形にする。質問や依頼だけの発言からは、何も抜き出さない。"
    ),
})


#####################################
# 実行時に渡す情報：誰との会話か
#####################################
@dataclass
class UserContext:
    user_id: str




#####################################
# ミドルウェア1：思い出す
#####################################
@dynamic_prompt
def add_memories_to_prompt(request: ModelRequest) -> str:
    question = request.messages[-1].text
    # mem0 で、この利用者の記憶だけを、質問に近い順に検索する（filters の user_id で、利用者ごとに分ける）
    items = memory.search(question, filters={"user_id": request.runtime.context.user_id}, top_k=3)["results"]
    memories = "\n".join(f"- {item['memory']}" for item in items) or "（なし）"
    print(f"  [思い出した記憶] {[item['memory'] for item in items]}")
    return (
        "あなたは利用者専属のアシスタントです。日本語で簡潔に答えてください。\n"
        "利用者について分かっていることは、次の記憶だけです。"
        "記憶に無いことは、知らないと答えてください。\n"
        f"【記憶】\n{memories}"
    )


#####################################
# ミドルウェア2：記憶する
#####################################
# 01_LangChain/02_07_MemoryUserProfile.py では、事実を抜き出すチェーンを自分で書き、既存の記憶と重ならないものを保存した。
# mem0 の add は、発言から事実を抜き出し、既存の記憶と比べて、追加（ADD）・更新（UPDATE）・削除（DELETE）まで自動で行う
@after_agent
def save_new_facts(state: AgentState, runtime) -> None:
    user_message = [m for m in state["messages"] if m.type == "human"][-1].text
    result_add = memory.add([{"role": "user", "content": user_message}], user_id=runtime.context.user_id)
    for item in result_add["results"]:
        print(f"  [記憶した（{item.get('event', 'ADD')}）] {item['memory']}")
    return None


#####################################
# エージェント
#####################################
agent = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール
    tools=[],
    # middleware    : 応答の前後に実行する処理（思い出す・記憶する）
    middleware=[add_memories_to_prompt, save_new_facts],
    # context_schema: 実行時に渡す情報の形（誰との会話か）
    context_schema=UserContext,
    # checkpointer  : 会話ごとの履歴の保存先（02_05と同じ）
    checkpointer=InMemorySaver(),
)


def main():
    """
    題材：
        AIアシスタントが、利用者から聞いた名前・好み・予定を記憶し、別の会話で思い出して使う。
        記憶の保存・検索と、発言から事実を抜き出す処理には、mem0 を使う。

    登場人物：
      AIアシスタント             ：このプログラムのエージェント。田中さん・鈴木さんとの会話を区別して記憶する
      田中さん（利用者ID：tanaka）：田中さんは、AIアシスタントへ、自分の名前・説明の好み・旅行の予定を伝える
      鈴木さん（利用者ID：suzuki）：鈴木さんは、AIアシスタントへ、自分の名前・説明の好み・旅行の予定を伝えない

    発言（5件）：
        会話1（thread-1）：田中さんが、名前・説明の好み・コード例の言語・旅行の予定を伝える。
        会話2（thread-2）：田中さんが、会話1とは別の会話で、好みや予定を覚えているかを尋ねる。
        会話3（thread-3）：鈴木さんが、旅行の予定を尋ねる。

    記憶の仕組み：
        思い出す（add_memories_to_prompt）：
            LLMを呼ぶ前に、memory.search で、利用者の記憶を質問に近い順に3件検索し、指示文に入れる。
            filters の user_id で、利用者ごとに記憶を分ける（鈴木さんには、田中さんの記憶は出ない）。
        記憶する（save_new_facts）：
            回答の後に、memory.add に利用者の発言を渡す。mem0 が、発言から事実を抜き出し、既存の記憶と比べて、
            追加（ADD）・更新（UPDATE）・削除（DELETE）を行う。抜き出し方は custom_instructions で指示する。

    流れ（発言1件分）：
        1. 思い出す：利用者の記憶を検索して、指示文に入れる。
        2. エージェントが、記憶を使って答える。
        3. 記憶する：発言から事実を抜き出して、保存する。
        すべての発言が終わったら、田中さんの記憶を「説明の好み」で検索し、近い順に表示する。

    01_LangChain/02_07_MemoryUserProfile.py との違い：
        記憶の保存先と、事実の抜き出し方が違う。
        01_LangChain は、保存先に InMemoryStore を使い、事実を抜き出すチェーンを自分で書き、既存の記憶と重ならないものだけを保存する。
        こちらは、保存・検索・抜き出し・既存の記憶との比較を、mem0 がまとめて行う。
        登場人物・発言・エージェントは同じ。

    注意：
        埋め込みベクトルは OpenAI にしかない機能のため、LLM_PROVIDER=openai で実行する。
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
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
            context=UserContext(user_id=user_id),
        )
        print(f"[{user_id} / {thread_id}] AI：{result['messages'][-1].text}")

    # 田中さんの記憶を、意味で直接検索する（「説明の好み」に近いものから並ぶ）
    print("--- 田中さんの記憶を「説明の好み」で検索 ---")
    for item in memory.search("説明の好み", filters={"user_id": "tanaka"}, top_k=5)["results"]:
        print(f"  類似度 {item['score']:.2f}：{item['memory']}")


if __name__ == "__main__":
    main()
