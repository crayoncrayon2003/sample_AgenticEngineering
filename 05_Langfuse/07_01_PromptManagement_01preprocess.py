import os

from dotenv import load_dotenv
from langfuse import get_client

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))

#####################################
# Langfuse：実行の記録（トレース）の送信先
#####################################
# ログの送信先を省略すると Langfuse Cloud（インターネット上）に送られる。ローカルの Langfuse に送信する
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

#####################################
# プロンプトを Langfuse に登録する
#####################################
# 01_LangChain/01_01_Chain.py では、プロンプトをPythonのコードの中に書いた。プロンプトを直すたびに、コードの変更が必要になる。
# Langfuse の「プロンプト管理」では、プロンプトをサーバーに保存し、バージョンとラベルで管理する。
#   バージョン：登録するたびに 1, 2, 3 … と増える。過去の版も残る
#   ラベル    ：版に付ける名札。production（本番で使う版）や staging（試している版）など。
#              画面でラベルを別の版に付け替えるだけで、コードを変えずにアプリが使うプロンプトを切り替えられる
# Langfuse のプロンプトでは、変数を {{review}} のように二重の波かっこで書く
PROMPT_NAME = "review_extract"

PROMPT_V1 = [
    {"role": "system", "content": "あなたは商品レビューを整理する係です。"},
    {"role": "user", "content": (
        "次の商品レビューから、良い点と気になる点を箇条書きで抜き出してください。\n"
        "レビューに書かれていないことは追加しないでください。\n\n{{review}}"
    )},
]
PROMPT_V2 = [
    {"role": "system", "content": "あなたは商品レビューを整理する係です。"},
    {"role": "user", "content": (
        "次の商品レビューから、良い点と気になる点を箇条書きで抜き出してください。\n"
        "気になる点には、商品の改善につながる提案を1つずつ添えてください。\n"
        "レビューに書かれていないことは追加しないでください。\n\n{{review}}"
    )},
]


def register_prompts() -> None:
    """プロンプトが未登録なら、版1（production）と版2（staging）を登録する。"""
    try:
        # 登録済みなら、何もしない（実行するたびに版が増えないようにする）
        langfuse.get_prompt(PROMPT_NAME, label="production", type="chat", cache_ttl_seconds=0)
        print(f"プロンプト「{PROMPT_NAME}」は登録済みです。")
        return
    except Exception:
        pass
    langfuse.create_prompt(name=PROMPT_NAME, type="chat", prompt=PROMPT_V1, labels=["production"],
                           commit_message="最初の版：良い点と気になる点を抜き出す")
    langfuse.create_prompt(name=PROMPT_NAME, type="chat", prompt=PROMPT_V2, labels=["staging"],
                           commit_message="改善の提案を添える")
    print(f"プロンプト「{PROMPT_NAME}」の版1（production）と版2（staging）を登録しました。")


def main():
    """
    題材：商品レビューから良い点・気になる点を抜き出すプロンプトを、Langfuse に登録する（準備）。
         登録したプロンプトは、07_01_PromptManagement_02process.py が取り出して使う。

    流れ：
        1. 07_01_PromptManagement_01preprocess.py（手元のPC）が、2つの版のプロンプトを Langfuse（サーバ）に登録する。
           版1に production、版2に staging のラベルを付ける。
        2. すでに登録済みなら、何もしない（何度実行しても、版は増えない）。
    """

    register_prompts()
    print(f"プロンプトの画面：{os.environ['LANGFUSE_BASE_URL']}/project/sample-project/prompts/{PROMPT_NAME}")


if __name__ == "__main__":
    main()
