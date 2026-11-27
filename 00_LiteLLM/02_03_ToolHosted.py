import os
import sys

import litellm
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")


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


# 組み込みツールは、LiteLLM の responses（OpenAI の Responses API と同じ形）で使う。
# このサンプルは OpenAI の組み込みツールを使うため、OpenAI 専用
if PROVIDER != "openai":
    sys.exit("このサンプルは OpenAI の組み込みツールを使うため、LLM_PROVIDER=openai で実行してください。")

llm = make_llm("low")

#####################################
# 組み込みツール
#####################################
# 関数ではなく辞書で指定する。実際の処理は、提供元のサーバーで行われる
tool_web_search = {"type": "web_search"}
tool_code_execution = {"type": "code_interpreter", "container": {"type": "auto"}}


#####################################
# 質問して、回答の中身を表示する
#####################################
# 組み込みツールを使って質問し、回答を表示して返す
def ask(tool: dict, instructions: str, question: str):
    print(f"質問：{question}")
    response = litellm.responses(
        model=llm["model"],
        reasoning={"effort": llm["reasoning_effort"]},
        instructions=instructions,
        input=question,
        tools=[tool],
        timeout=180,
    )
    # output には、ツールの実行記録と、最後の回答（message）が順に入っている
    answer = response.output[-1]
    print(f"回答：{answer.content[0].text}")
    return response


# 回答に付いた出典（URLやファイル名）を表示する
def show_citations(response) -> None:
    for citation in response.output[-1].content[0].annotations or []:
        print(f"  出典：{getattr(citation, 'title', '') or getattr(citation, 'filename', '')} {getattr(citation, 'url', '')}")


# 提供元のサーバーで実行されたコードを表示する
def show_executed_code(response) -> None:
    for item in response.output:
        if item.type == "code_interpreter_call":
            print("  [実行されたコード]")
            print(item.code)


def main():
    """
    題材：提供元の組み込みツールを使う3つの例（OpenAI専用）。
         1. Web検索      ：出典付きで最近の話題を答える
         2. コード実行   ：提供元のサーバーでPythonを実行して計算する
         3. ファイル検索 ：OpenAIに登録した文書を検索して答える
    """

    print("========== 1. Web検索 ==========")
    response = ask(tool_web_search, "必ずWeb検索を使い、一次資料を優先して、日付と出典付きで答えてください。",
                   "Pythonの最新の安定版のバージョンと、そのリリース日を教えてください。")
    show_citations(response)

    print("========== 2. コード実行 ==========")
    response = ask(tool_code_execution, "計算は必ずコード実行ツールでPythonを実行して行い、結果を簡潔に答えてください。",
                   "30の階乗の各桁の数字を合計してください。")
    show_executed_code(response)

    print("========== 3. ファイル検索 ==========")
    if not VECTOR_STORE_ID:
        print(".env に OPENAI_VECTOR_STORE_ID が無いため、実行できません。")
        return
    tool_file_search = {"type": "file_search", "vector_store_ids": [VECTOR_STORE_ID]}
    # 2件目は、文書に答えが書かれていない質問
    for question in ["第2四半期の売上目標は？", "研究室Xの電話番号は？"]:
        response = ask(tool_file_search, "必ずfile_searchで調べ、見つからない場合は『検索結果では確認できませんでした』と答えてください。",
                       question)
        show_citations(response)


if __name__ == "__main__":
    main()
