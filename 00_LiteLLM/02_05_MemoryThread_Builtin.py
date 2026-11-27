import os

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


llm = make_llm("minimal")


#####################################
# 会話履歴の保存先（LangChain の checkpointer に当たる）
#####################################
# 保存先の切り替え（sqlite・postgres）は 02_05_MemoryThread.py を参照。ここでは要約の動きに絞り、メモリに保存する
history_store = {}


#####################################
# 会話履歴が長くなったら、古い部分を要約する（LangChain の SummarizationMiddleware に当たる）
#####################################
# 02_05_MemoryThread.py は、会話履歴はすべて残り、質問するたびに増え続ける。
# 02_05_MemoryThread_Builtin.pyは、履歴が MAX_MESSAGES を超えたら、古いメッセージを1つの要約に置き換える。
# 残すメッセージは、KEEP_MESSAGES の数だけである。LLMに渡すメッセージ量が増え続けるのを防げぐ。
MAX_MESSAGES   = 6  # 要約を始める条件。メッセージがMAX_MESSAGES件を超えたら要約する
KEEP_MESSAGES  = 2  # 要約せずに、そのまま残す新しいメッセージの数
SUMMARY_PREFIX = "これまでの会話の要約："

summary_prompt = (
    "あなたは会話の要約係です。次の会話から、今後の会話に必要な事実（利用者の好み、決めたことなど）を落とさず、"
    "日本語で簡潔に要約してください。"
)


def summarize(messages: list) -> str:
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return call_llm(llm, [
        {"role": "system", "content": summary_prompt},
        {"role": "user",   "content": conversation},
    ]).content


def compress_history(history: list) -> list:
    if len(history) <= MAX_MESSAGES:
        return history
    old, keep = history[:-KEEP_MESSAGES], history[-KEEP_MESSAGES:]
    summary = summarize(old)
    # 要約は、履歴の先頭に1件のメッセージとして入れる
    return [{"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"}] + keep


#####################################
# エージェント（会話履歴を、自分で読み出し、要約して、保存する）
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = "あなたは親切なアシスタントです。日本語で1〜2文で答えてください。"


def chat(message: str, thread_id: str) -> dict:
    # 同じ thread_id で、前回保存した会話履歴を読み出す（初回は空）
    history = history_store.get(thread_id, [])
    history.append({"role": "user", "content": message})

    # LLMを呼ぶ前に、長くなった履歴を要約する
    history = compress_history(history)

    answer = call_llm(llm, [{"role": "system", "content": system_prompt}] + history).content
    history.append({"role": "assistant", "content": answer})

    # 次回のために、会話履歴を保存する
    history_store[thread_id] = history
    return {"answer": answer, "history": history}


def main():
    """
    題材：1つの会話（thread-A）で5回話しかけ、会話履歴の件数の変化を見る。
         1回目に好きな食べ物を伝え、2〜4回目は別の話題、5回目に好きな食べ物を確認する。
         履歴が6件を超えると古い部分が要約に置き換わるため、件数は増え続けない。
         それでも、要約に残った内容から、1回目に伝えた好きな食べ物を答えられるかを確かめる。
    """

    messages = [
        "私の好きな食べ物はカレーです。",
        "今日はいい天気ですね。",
        "おすすめの散歩コースの選び方を教えてください。",
        "雨の日の過ごし方も教えてください。",
        "私の好きな食べ物は何でしたか？",
    ]
    for message in messages:
        print("----------------------------------------")
        print(f"[thread-A] 利用者：{message}")
        result = chat(message, "thread-A")
        print(f"[thread-A] AI：{result['answer']}")

        # 保存されている会話履歴の件数と、要約が含まれているかを表示する
        history = result["history"]
        summaries = [m for m in history if m["content"].startswith(SUMMARY_PREFIX)]
        print(f"  （保存済みメッセージ {len(history)}件{'、うち1件は要約' if summaries else ''}）")
        if summaries:
            body = summaries[0]["content"].removeprefix(SUMMARY_PREFIX).strip()
            print(f"  要約の内容（先頭150文字）：{body[:150].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
