import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
# Disable DeepEval telemetry before importing DeepEval.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
EVAL_DIR = os.path.join(ROOT, "outputs", "evaluations")


def make_llm(effort: str):
    """Create an OpenAI or Claude chat model based on LLM_PROVIDER. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
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
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


#####################################
# LLM used for DeepEval grading
#####################################
# DeepEval uses LLMs for metrics such as GEval. Without configuration, it calls OpenAI directly.
# Wrap a LangChain model from make_llm in a DeepEvalBaseLLM subclass to support LLM_PROVIDER selection.
class LangChainModel(DeepEvalBaseLLM):
    def __init__(self, model):
        self.chat_model = model
        super().__init__(model="langchain")

    def load_model(self):
        return self.chat_model

    # schema: Return the requested structure when DeepEval supplies a Pydantic schema.
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
# Two instruction sets to compare (A: Simple; B: Detailed)
#####################################
INSTRUCTIONS = {
    "A": 'You explain technology. Answer beginners in English in about 200 characters.',
    "B": 'You explain technology. Answer beginners in English in about 200 characters. Start with the key point, explain abbreviations, and include one concrete example. If asked to compare, make the differences clear.',
}

#####################################
# Test cases (questions and reference facts for grading)
#####################################
TEST_CASES = [
    {
        "question": 'Explain the difference between ETL and ELT with a concrete example.',
        "reference": 'ETL means extract, transform, load; ELT means extract, load, transform. The difference is whether transformation happens before or after loading.',
    },
    {
        "question": 'Explain the different roles of MCP and A2A in AI agents with concrete examples.',
        "reference": 'MCP connects AI applications to external tools and data. A2A enables agents to exchange requests and results.',
    },
]

#####################################
# Answer generation chain
#####################################
prompt_answer = ChatPromptTemplate.from_messages([
    # system: Instructions from the application developer (A or B)
    ("system", "{instruction}"),
    # user: A message from the application user
    ("user", "{question}"),
])
chain_answer = prompt_answer | llm | StrOutputParser()


#####################################
# Grading result schema
#####################################
class JudgeResult(BaseModel):
    # Each score is 0-1 (1 is highest), the range returned by DeepEval GEval.
    reason: str
    correctness: float
    clarity: float
    completeness: float


#####################################
# Grade using an LLM judge through DeepEval GEval
#####################################
# GEval receives textual criteria, has the LLM devise evaluation steps, then grades on a 0-1 scale.
# evaluation_params: Inputs used for grading (question, answer, reference facts)
judge_model = LangChainModel(llm)
JUDGE_PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]
metrics = {
    "correctness": GEval(name="correctness", criteria='Whether the answer is accurate and consistent with the reference facts (expected output).', evaluation_params=JUDGE_PARAMS, model=judge_model, async_mode=False),
    "clarity": GEval(name="clarity", criteria='Whether beginners can understand the explanation. Do not award higher scores merely for length or preference.', evaluation_params=JUDGE_PARAMS, model=judge_model, async_mode=False),
    "completeness": GEval(name="completeness", criteria='Whether the answer includes the requested differences and concrete examples.', evaluation_params=JUDGE_PARAMS, model=judge_model, async_mode=False),
}


def judge(inputs: dict) -> JudgeResult:
    test_case = LLMTestCase(input=inputs["question"], actual_output=inputs["answer"], expected_output=inputs["reference"])
    scores = {name: round(metric.measure(test_case), 2) for name, metric in metrics.items()}
    return JudgeResult(reason=metrics["correctness"].reason, **scores)

def main():
    """Example: Compare which of two instruction sets yields better technical answers. Unlike 03_08, text answers have no single correct output, so use an LLM judge rather than exact matching.
    A requests about 200 characters for beginners. B additionally requires key points first, abbreviation explanations, one example, and clear differences. Cases: ETL vs ELT (extract-transform-load vs extract-load-transform), and MCP vs A2A (external tools/data vs agent-to-agent exchanges). Two instruction sets times two questions yield four answers.
    judge uses GEval for correctness, clarity, and completeness. GEval receives textual criteria, devises steps, and grades with an LLM on a 0-1 scale (0 not met, 1 fully met). Supply the same model as generation via make_llm wrapped in LangChainModel. Compare per-instruction mean scores without pass/fail decisions (the original text describes four answers times three criteria per instruction set).
    For each answer: Generate, grade through LLMTestCase, and display. Finally display means and save JSON in outputs/evaluations/. Unlike 01_LangChain/03_09_EvalJudge.py's custom 1-5 grading prompt, use GEval with 0-1 scores; cases, generation, flow, and saving are unchanged. GEval's grading instructions are English, so reasons may be in English.
    """

    records = []
    for case in TEST_CASES:
        for name, instruction in INSTRUCTIONS.items():
            print("----------------------------------------")
            print(f"Instructions: {name}; question: {case['question']}")

            # Stage 1: Generate an answer
            result_step1 = chain_answer.invoke({"instruction": instruction, "question": case["question"]})
            print(f'Answer: {result_step1}')

            # Stage 2: Grade the answer
            result_step2 = judge({"question": case["question"], "reference": case["reference"], "answer": result_step1})
            scores = {"correctness": result_step2.correctness, "clarity": result_step2.clarity, "completeness": result_step2.completeness}
            print(f'Scores: {scores} (reason: {result_step2.reason})')

            records.append({"instruction": name, "question": case["question"], "answer": result_step1, "scores": scores, "reason": result_step2.reason})

    # Compare average scores by instruction set.
    print("----------------------------------------")
    summary = {}
    for name in INSTRUCTIONS:
        all_scores = [score for record in records if record["instruction"] == name for score in record["scores"].values()]
        summary[name] = round(sum(all_scores) / len(all_scores), 2)
        print(f'Mean scores for instructions {name}: {summary[name]}')

    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"judge_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "records": records}, file, ensure_ascii=False, indent=2)
    print(f'Saved to: {path}')


if __name__ == "__main__":
    main()
