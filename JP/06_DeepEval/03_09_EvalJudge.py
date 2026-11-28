import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
# DeepEval の利用状況の送信（テレメトリ）を止める。DeepEval を import する前に設定する
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

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
# DeepEval の採点に使うLLM
#####################################
# DeepEval は、採点（GEval など）にLLMを使う。何も指定しないと OpenAI を直接呼ぶ。
# ここでは make_llm で作った LangChain のモデルを使うため、DeepEvalBaseLLM を継承した部品で包む（LLM_PROVIDER で切り替えられる）
class LangChainModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.chat_model = model
        super().__init__(model="langchain")

    def load_model(self):
        return self.chat_model

    # schema：DeepEval が回答の形（Pydantic のクラス）を指定したときは、その形で返す
    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.chat_model.with_structured_output(schema, method="json_schema").invoke(prompt)
        return self.chat_model.invoke(prompt).text

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self.chat_model.with_structured_output(schema, method="json_schema").ainvoke(prompt)
        return (await self.chat_model.ainvoke(prompt)).text

    def get_model_name(self) -> str:
        return f"LangChain（{PROVIDER}）"


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
    #  各項目は0〜1点（1点が最高）。DeepEval の GEval は、点数を0〜1の範囲で返す
    reason: str
    correctness: float
    clarity: float
    completeness: float


#####################################
# 採点する（DeepEval の GEval で、LLMを採点者として使う）
#####################################
# GEval：採点の基準（criteria）を文章で渡すと、LLMが採点の手順を考えてから、0〜1点で採点する。
#   evaluation_params：採点に使う項目（質問・回答・参考事項）
judge_model = LangChainModel(llm)
JUDGE_PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]
metrics = {
    "correctness": GEval(name="correctness", criteria="回答の内容が、参考事項（expected output）と矛盾せず正確か。", evaluation_params=JUDGE_PARAMS, model=judge_model, async_mode=False),
    "clarity": GEval(name="clarity", criteria="初学者にとって、説明が分かりやすいか。文章の長さや好みだけで点を上げない。", evaluation_params=JUDGE_PARAMS, model=judge_model, async_mode=False),
    "completeness": GEval(name="completeness", criteria="質問が求める違いと具体例を、回答が含んでいるか。", evaluation_params=JUDGE_PARAMS, model=judge_model, async_mode=False),
}


def judge(inputs: dict) -> JudgeResult:
    test_case = LLMTestCase(input=inputs["question"], actual_output=inputs["answer"], expected_output=inputs["reference"])
    scores = {name: round(metric.measure(test_case), 2) for name, metric in metrics.items()}
    return JudgeResult(reason=metrics["correctness"].reason, **scores)

def main():
    """
    題材：
        2種類の指示文（A・B）のうち、どちらが良い回答を引き出せるかを、テストで比べる。
        03_08 と違い、回答は文章のため、正解が1つに決まらない。完全一致では判定できないため、LLMに採点させる。

    評価の対象：
        回答を作るLLMが、指示文（A か B）に従って、技術の質問に答える。
          A：簡単な指示（初学者向けに、200字程度で答える）
          B：詳しい指示（A に加えて、要点を先に述べる・略語を説明する・具体例を1つ含める・違いを明確にする）

    テストケース（2件）：
        質問と、採点の参考にする正しい内容（参考事項）の組。
          1. ETLとELTの違い      ：参考事項「ETLは抽出・変換・格納、ELTは抽出・格納・変換の順」
          2. MCPとA2Aの役割の違い：参考事項「MCPは外部のツール・データとつなぐ、A2Aはエージェント同士がやり取りする」
        指示文2種類 × 質問文2件 ＝ 4つの回答を採点する。

    判定すること（judge）：
        採点を担当するLLMに、3つの基準で採点させる。判定には、DeepEval の GEval 関数を使う。
          1. correctness ：回答の内容が、参考事項と矛盾せず正確か
          2. clarity     ：初学者にとって、説明が分かりやすいか
          3. completeness：質問が求める違いと具体例を、回答が含んでいるか
        GEval は、基準（criteria）を文章で受け取り、採点の手順を考えてから、LLMに採点させる。
        点数は、0〜1 の範囲である。0 は「まったく満たしていない」、1 は「完全に満たしている」に該当する。
        採点を担当するLLMは、make_llm で作ったモデルを LangChainModel で包んで渡す（回答を作るLLMと同じモデル）。

    比べ方：
        合格・不合格は判定しない。指示文ごとに、4つの回答 × 3つの基準の点数を平均し、平均点を比べる。

    流れ（回答1つ分）：
        1. 回答を作るLLMが、指示文と質問から回答を作る。
        2. judge が、質問・回答・参考事項を、DeepEval のテストケース（LLMTestCase）に入れて、3つの基準で採点する。
        3. 結果を表示する。
        すべての回答が終わったら、指示文ごとの平均点を表示し、結果を outputs/evaluations/ の JSON ファイルに保存する。

    01_LangChain/03_09_EvalJudge.py との違い：
        採点（judge）の中身だけが違う。01_LangChain は採点用のプロンプトを自分で書き、1〜5点で採点させる。
        こちらは DeepEval の GEval で採点し、点数は 0〜1 になる。
        テストケース・回答の作り方・流れ・結果の保存は同じ。

    注意：
        GEval が返す採点の理由は、DeepEval の採点用の指示が英語のため、英語になることがある。
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
