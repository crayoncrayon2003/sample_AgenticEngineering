import math
import os
import sys

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic

# 埋め込みベクトルは OpenAI にしかない機能のため、Claude では実行できない
if PROVIDER != "openai":
    sys.exit("このサンプルは埋め込みベクトル（OpenAI専用）を使うため、LLM_PROVIDER=openai で実行してください。")


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


llm = make_llm("minimal")

#####################################
# 長期記憶の保存場所（LangChain の store に当たる）
#####################################
# 利用者ID ごとに、記憶（文章と、その埋め込みベクトル）の一覧を保存する
memory_store = {}


# 文章を埋め込みベクトル（数値の並び）に変える
def embed(text: str) -> list[float]:
    response = litellm.embedding(model="openai/text-embedding-3-small", input=[text])
    return response.data[0]["embedding"]


# 2つのベクトルの近さ（コサイン類似度）を計算する
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# 利用者の記憶を、意味の近い順に検索する
def search_memories(user_id: str, query: str, limit: int) -> list[dict]:
    query_vector = embed(query)
    items = []
    for memory in memory_store.get(user_id, []):
        items.append({"text": memory["text"], "score": cosine_similarity(query_vector, memory["vector"])})
    items.sort(key=lambda item: item["score"], reverse=True)
    return items[:limit]


# 利用者の記憶を1件保存する
def save_memory(user_id: str, text: str) -> None:
    memory_store.setdefault(user_id, []).append({"text": text, "vector": embed(text)})


#####################################
# 記憶を抜き出す処理
#####################################
# LLMのレスポンス形式を定義する（抜き出した事実）
class NewFacts(BaseModel):
    #  facts： 利用者の発言から抜き出した、利用者自身についての事実の一覧
    facts: list[str] = Field(
        description="利用者の名前・好み・予定など、今後の会話でも役立つ情報。"
        "1件1文で「利用者は〜」の形にする。質問だけで新しい情報が無ければ空のリスト"
    )


prompt_extract = [
    # system : アプリを作った開発者からの指示
    {"role": "system", "content": (
        "利用者の発言から、利用者自身についての新しい事実だけを抜き出してください。\n"
        "- 質問や依頼だけの発言からは、何も抜き出さない（空のリスト）。\n"
        "- 既存の記憶と同じ内容は、抜き出さない。\n"
        "例：\n"
        "  「佐藤です。猫を飼っています。」→ [\"利用者の名前は佐藤。\", \"利用者は猫を飼っている。\"]\n"
        "  「私の名前を覚えていますか？」→ []"
    )},
    # user   : アプリを使う利用者の発言
    {"role": "user", "content": "【既存の記憶】{known}\n【利用者の発言】{message}"},
]
# 抜き出しは判断が難しいため、推論を少し深くしたモデルを使う
llm_extract = make_llm("low")


# LangChain の chain_extract.invoke({...}) に当たる。プロンプトに入れる値を、辞書で受け取る
def chain_extract(values: dict) -> NewFacts:
    response = call_llm(llm_extract, format_prompt(prompt_extract, values), response_format=NewFacts)
    return NewFacts.model_validate_json(response.content)


# 記憶を入れたシステムプロンプトを作る
def make_system_prompt(memories: str) -> str:
    return (
        "あなたは利用者専属のアシスタントです。日本語で簡潔に答えてください。\n"
        "利用者について分かっていることは、次の記憶だけです。"
        "記憶に無いことは、知らないと答えてください。\n"
        f"【記憶】\n{memories}"
    )


#####################################
# エージェント：思い出す → 回答する → 記憶する
#####################################
# 会話ごとの履歴
threads = {}


def run_agent(message: str, user_id: str, thread_id: str) -> str:
    history = threads.get(thread_id, [])
    history.append({"role": "user", "content": message})

    # Step1 思い出す：発言に近い記憶を検索する（LangChain では、ミドルウェア dynamic_prompt で行っていた）
    items = search_memories(user_id, message, limit=3)
    memories = "\n".join(f"- {item['text']}" for item in items) or "（なし）"
    print(f"  [思い出した記憶] {[item['text'] for item in items]}")

    # Step2 回答する
    answer = call_llm(llm, [{"role": "system", "content": make_system_prompt(memories)}] + history).content
    history.append({"role": "assistant", "content": answer})
    threads[thread_id] = history

    # Step3 記憶する：発言から事実を抜き出して保存する（LangChain では、ミドルウェア after_agent で行っていた）
    known = [memory["text"] for memory in memory_store.get(user_id, [])]
    result_extract = chain_extract({"known": known, "message": message})
    for fact in result_extract.facts:
        save_memory(user_id, fact)
        print(f"  [記憶した] {fact}")
    return answer


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

        # エージェントを実行する
        answer = run_agent(message, user_id, thread_id)
        print(f"[{user_id} / {thread_id}] AI：{answer}")

    # 田中さんの記憶を、意味で直接検索する（「説明の好み」に近いものから並ぶ）
    print("--- 田中さんの記憶を「説明の好み」で検索 ---")
    for item in search_memories("tanaka", "説明の好み", limit=5):
        print(f"  類似度 {item['score']:.2f}：{item['text']}")


if __name__ == "__main__":
    main()
