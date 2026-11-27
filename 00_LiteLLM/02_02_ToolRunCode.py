import ast
import json
import os
import subprocess
import sys
from uuid import uuid4

import litellm
from dotenv import load_dotenv

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
CODE_DIR = os.path.join(ROOT, "outputs", "generated_code")
ALLOWED_IMPORTS = {"math", "datetime", "statistics", "itertools", "fractions", "decimal"}


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


llm = make_llm("low")


#####################################
# 生成コードの検査
#####################################
def check_code(code: str) -> None:
    """構文と、使われているimportを検査する。問題があれば ValueError を発生させる。"""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"使用できないモジュールです：{name}")


#########################################################
# ツール1 : Pythonコードをファイルに保存する
#########################################################
def save_code(code: str) -> str:
    """Pythonコードを.pyファイルに保存し、ファイル名を返す。"""
    try:
        check_code(code)
    except (SyntaxError, ValueError) as error:
        return f"保存できませんでした：{error}"

    os.makedirs(CODE_DIR, exist_ok=True)
    filename = f"calc_{uuid4().hex[:8]}.py"
    with open(os.path.join(CODE_DIR, filename), "w", encoding="utf-8") as file:
        file.write(code + "\n")
    print(f"[保存] {filename}")
    print(code)
    return filename

#########################################################
# ツール2 : 指定パスにあるPythonコードを実行する
#########################################################
def run_code(filename: str) -> str:
    """save_code が返したファイル名のPythonコードを実行し、標準出力を返す。"""
    path = os.path.realpath(os.path.join(CODE_DIR, filename))
    # 保存フォルダー以外のファイルは実行しない
    if os.path.dirname(path) != os.path.realpath(CODE_DIR) or not os.path.isfile(path):
        return "実行できませんでした：save_code が返したファイル名を指定してください。"

    try:
        completed = subprocess.run(
            [sys.executable, "-I", path],  # -I：環境変数や作業フォルダーの影響を受けない
            cwd=CODE_DIR,
            # APIキーなどの環境変数を渡さない（SYSTEMROOTはWindowsでの起動に必要）
            env={k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT"}},
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "実行できませんでした：10秒以内に終わりませんでした。"

    if completed.returncode != 0:
        return f"実行エラー：\n{completed.stderr.strip()}"
    print(f"[実行結果] {completed.stdout.strip()}")
    return completed.stdout.strip()


#####################################
# LLMに渡すツールの説明
#####################################
tools = [
    {"type": "function", "function": {
        "name": "save_code",
        "description": "Pythonコードを.pyファイルに保存し、ファイル名を返す。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "保存するPythonコード全体。説明文やコードフェンスは含めない。"}},
            "required": ["code"],
        },
    }},
    {"type": "function", "function": {
        "name": "run_code",
        "description": "save_code が返したファイル名のPythonコードを実行し、標準出力を返す。",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "save_code が返したファイル名"}},
            "required": ["filename"],
        },
    }},
]
# ツール名から、実行する関数を探すための対応表
tool_functions = {"save_code": save_code, "run_code": run_code}


#####################################
# エージェント
#####################################
system_prompt = """
あなたは計算担当です。計算は暗算せず、必ず次の手順で行ってください。
1. 計算結果をprintするPythonコードを書き、save_codeで保存する。
2. 返されたファイル名をrun_codeに渡して実行する。
3. 実行結果に基づいて、日本語で簡潔に答える。

ルール：
- 使えるimportは {allowed_imports} だけです。
- 失敗した場合は1回だけ直して再実行し、それでも失敗したら理由を伝えてください。
"""


def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt.format(allowed_imports=", ".join(sorted(ALLOWED_IMPORTS)))},
        {"role": "user", "content": question},
    ]
    message = call_llm(llm, messages, tools=tools)

    # LLMがツール呼び出しを返す間は、ツールを実行して、もう一度LLMを呼ぶ
    while message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = tool_functions[tool_call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        message = call_llm(llm, messages, tools=tools)

    return message.content


def main():
    """
    題材：計算問題を解く。解答するために専用のpythonコードを生成・実行する。
         質問は、合計2件ある。
    """

    # 質問の一覧
    questions = [
        "1から1000までの素数の合計を求めてください。",
        "2026年4月1日から2026年12月25日まで、何日ありますか？",
    ]
    for question in questions:
        print("----------------------------------------")
        print(f"質問：{question}")
        # エージェントを実行する
        answer = run_agent(question)
        print(f"回答：{answer}")


if __name__ == "__main__":
    main()
