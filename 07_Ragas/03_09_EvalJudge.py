import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
# Ragas の利用状況の送信（テレメトリ）を止める。Ragas を import する前に設定する
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
from ragas.llms import llm_factory
from ragas.metrics.collections import DomainSpecificRubrics

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
EVAL_DIR = os.path.join(ROOT, "outputs", "evaluations")


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


#####################################
# Ragas の評価に使うLLM
#####################################
# Ragas の評価指標は、LangChain のモデルではなく、Ragas 用のLLM（llm_factory で作る）を使う。LLM_PROVIDER に応じて作り分ける
#   max_tokens：推論するモデルは、考える分のトークンも使うため、多めにしておく（足りないと評価が途中で止まる）
def make_ragas_llm():
    if PROVIDER == "openai":
        from openai import OpenAI

        return llm_factory(os.getenv("OPENAI_MODEL") or "gpt-5-nano", client=OpenAI(), max_tokens=16000)
    from anthropic import Anthropic

    return llm_factory(os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5", provider="anthropic", client=Anthropic(), max_tokens=16000)


llm = make_llm("minimal")

#####################################
# 比べる2種類の指示文（A：簡単な指示、B：詳しい指示）
#####################################
INSTRUCTIONS = {
    "A": "技術解説の係です。質問に、初学者向けに日本語で200字程度で答えてください。",
    "B": "技術解説の係です。質問に、初学者向けに日本語で200字程度で答えてください。最初に要点を述べ、略語は意味を説明し、具体例を1つ含めてください。比較を求められたら、違いを明確にしてください。",
}

#####################################
# テストケース（質問と、採点の参考にする正しい内容）
#####################################
TEST_CASES = [
    {
        "question": "ETLとELTの違いを、具体例を含めて説明してください。",
        "reference": "ETLは抽出・変換・格納、ELTは抽出・格納・変換の順。違いは、格納の前後どちらで変換するか。",
    },
    {
        "question": "AIエージェントで使うMCPとA2Aの役割の違いを、具体例を含めて説明してください。",
        "reference": "MCPはAIアプリと外部のツール・データをつなぐプロトコル。A2Aはエージェント同士が依頼や結果をやり取りするプロトコル。",
    },
]

#####################################
# 回答を作るチェーン
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    # system : アプリを作った開発者からの指示（A か B）
    ("system", "{instruction}"),
    # user   : アプリを使う利用者の発言
    ("user", "{question}"),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# 採点結果の形
#####################################
class JudgeResult(BaseModel):
    #  各項目は1〜5点（5点が最高）
    reason: str
    correctness: int
    clarity: int
    completeness: int


#####################################
# 採点する（Ragas の DomainSpecificRubrics で、LLMを採点者として使う）
#####################################
# DomainSpecificRubrics：1〜5点それぞれの基準（ルーブリック）を文章で渡すと、LLMがその基準に照らして採点する。
#   with_reference=True：参考事項（reference）と比べて採点する
def make_rubrics(item: str) -> dict:
    return {
        "score1_description": f"{item}が、まったく満たされていない。",
        "score2_description": f"{item}が、ほとんど満たされていない。",
        "score3_description": f"{item}が、一部だけ満たされている。",
        "score4_description": f"{item}が、ほぼ満たされている。",
        "score5_description": f"{item}が、完全に満たされている。",
    }


ragas_llm = make_ragas_llm()
metrics = {
    "correctness": DomainSpecificRubrics(llm=ragas_llm, rubrics=make_rubrics("参考事項と矛盾しない、内容の正確さ"), with_reference=True),
    "clarity": DomainSpecificRubrics(llm=ragas_llm, rubrics=make_rubrics("初学者にとっての分かりやすさ（文章の長さや好みは含めない）"), with_reference=True),
    "completeness": DomainSpecificRubrics(llm=ragas_llm, rubrics=make_rubrics("質問が求める違いと具体例を含んでいること"), with_reference=True),
}


def judge(inputs: dict) -> JudgeResult:
    results = {name: metric.score(user_input=inputs["question"], response=inputs["answer"], reference=inputs["reference"])
               for name, metric in metrics.items()}
    return JudgeResult(reason=str(results["correctness"].reason), **{name: int(r.value) for name, r in results.items()})

def main():
    """
    題材：
        2種類の指示文（A・B）のうち、どちらが良い回答を引き出せるかを、テストで比べる。
        回答は文章のため、正解が1つに決まらない。完全一致では判定できないため、LLMに採点させる。

    評価の対象：
        回答を作るLLMが、指示文（A か B）に従って、技術の質問に答える。
          A：簡単な指示（初学者向けに、200字程度で答える）
          B：詳しい指示（A に加えて、要点を先に述べる・略語を説明する・具体例を1つ含める・違いを明確にする）

    テストケース（2件）：
        質問と、採点の参考にする正しい内容（参考事項）の組。
          1. ETLとELTの違い      ：参考事項「ETLは抽出・変換・格納、ELTは抽出・格納・変換の順」
          2. MCPとA2Aの役割の違い：参考事項「MCPは外部のツール・データとつなぐ、A2Aはエージェント同士がやり取りする」
        指示文2種類 × 質問2件 ＝ 4つの回答を採点する。

    判定すること（judge）：
        回答を、採点を担当するLLMに、3つの基準で採点させる。判定には、Ragas の DomainSpecificRubrics を使う。
          1. correctness ：回答の内容が、参考事項と矛盾せず正確か
          2. clarity     ：初学者にとって、説明が分かりやすいか
          3. completeness：質問が求める違いと具体例を、回答が含んでいるか
        DomainSpecificRubrics は、1〜5点それぞれの基準（ルーブリック）を文章で受け取り、LLMにその基準で採点させる。
        点数は、1〜5 の範囲である。1 は「まったく満たされていない」、5 は「完全に満たされている」に該当する。
        採点を担当するLLMは、make_ragas_llm で作る（Ragas は、LangChain のモデルではなく、Ragas 用のLLMを使う）。

    比べ方：
        合格・不合格は判定しない。指示文ごとに、2つの回答 × 3つの基準の点数を平均し、平均点を比べる。

    流れ（回答1つ分）：
        1. 回答を作るLLMが、指示文と質問から回答を作る。
        2. judge が、質問・回答・参考事項を、3つの評価指標に渡して採点する。
        3. 結果を表示する。
        すべての回答が終わったら、指示文ごとの平均点を表示し、結果を outputs/evaluations/ の JSON ファイルに保存する。

    01_LangChain/03_09_EvalJudge.py との違い：
        採点（judge）の中身だけが違う。01_LangChain は採点用のプロンプトを自分で書き、こちらは Ragas の評価指標で採点する。
        点数の範囲（1〜5点）・テストケース・回答の作り方・流れ・結果の保存は同じ。
    """

    records = []
    for case in TEST_CASES:
        for name, instruction in INSTRUCTIONS.items():
            print("----------------------------------------")
            print(f"指示文：{name}、質問：{case['question']}")

            # 1段目：回答を作る
            result_step1 = chain_answer.invoke({"instruction": instruction, "question": case["question"]})
            print(f"回答：{result_step1}")

            # 2段目：回答を採点する
            result_step2 = judge({"question": case["question"], "reference": case["reference"], "answer": result_step1})
            scores = {"correctness": result_step2.correctness, "clarity": result_step2.clarity, "completeness": result_step2.completeness}
            print(f"採点：{scores}（理由：{result_step2.reason}）")

            records.append({"instruction": name, "question": case["question"], "answer": result_step1, "scores": scores, "reason": result_step2.reason})

    # 指示文ごとに、平均点を比べる
    print("----------------------------------------")
    summary = {}
    for name in INSTRUCTIONS:
        all_scores = [score for record in records if record["instruction"] == name for score in record["scores"].values()]
        summary[name] = round(sum(all_scores) / len(all_scores), 2)
        print(f"指示文 {name} の平均点：{summary[name]}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"judge_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "records": records}, file, ensure_ascii=False, indent=2)
    print(f"保存先：{path}")


if __name__ == "__main__":
    main()
