import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
# Disable Ragas telemetry before importing Ragas.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
from ragas.llms import llm_factory
from ragas.metrics.collections import DomainSpecificRubrics

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
# LLM used for Ragas evaluation
#####################################
# Ragas metrics use a Ragas-specific LLM from llm_factory, rather than a LangChain model. Create one based on LLM_PROVIDER.
# max_tokens: Reasoning models also spend tokens on reasoning, so allow extra tokens to avoid interrupting evaluation.
def make_ragas_llm():
    if PROVIDER == "openai":
        from openai import OpenAI

        return llm_factory(os.getenv("OPENAI_MODEL") or "gpt-5-nano", client=OpenAI(), max_tokens=16000)
    from anthropic import Anthropic

    return llm_factory(os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5", provider="anthropic", client=Anthropic(), max_tokens=16000)


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
    # Each criterion is scored 1-5 (5 is highest).
    reason: str
    correctness: int
    clarity: int
    completeness: int


#####################################
# Grade with an LLM judge using Ragas DomainSpecificRubrics
#####################################
# DomainSpecificRubrics receives textual rubrics for each score from 1 to 5 and grades against them with an LLM.
# with_reference=True: Grade against reference facts (reference).
def make_rubrics(item: str) -> dict:
    return {
        "score1_description": f'{item} is not satisfied at all.',
        "score2_description": f'{item} is barely satisfied.',
        "score3_description": f'{item} is partially satisfied.',
        "score4_description": f'{item} is mostly satisfied.',
        "score5_description": f'{item} is fully satisfied.',
    }


ragas_llm = make_ragas_llm()
metrics = {
    "correctness": DomainSpecificRubrics(llm=ragas_llm, rubrics=make_rubrics('Factual accuracy consistent with reference facts'), with_reference=True),
    "clarity": DomainSpecificRubrics(llm=ragas_llm, rubrics=make_rubrics('Clarity for beginners (excluding text length and personal preference)'), with_reference=True),
    "completeness": DomainSpecificRubrics(llm=ragas_llm, rubrics=make_rubrics('Inclusion of the requested differences and concrete examples'), with_reference=True),
}


def judge(inputs: dict) -> JudgeResult:
    results = {name: metric.score(user_input=inputs["question"], response=inputs["answer"], reference=inputs["reference"])
               for name, metric in metrics.items()}
    return JudgeResult(reason=str(results["correctness"].reason), **{name: int(r.value) for name, r in results.items()})

def main():
    """Example: Compare answers generated by instruction sets A and B using LLM grading, as text has no single correct answer. A requests about 200 characters for beginners. B also requests key points first, abbreviation explanations, one example, and clear differences.
    Two cases pair questions with reference facts: ETL vs ELT (extract-transform-load vs extract-load-transform) and MCP vs A2A (tools/data connections vs agent exchanges). Two instructions times two questions produce four answers.
    judge uses Ragas DomainSpecificRubrics for correctness, clarity, and completeness. Supply textual rubrics for scores 1-5, where 1 means not met and 5 fully met. Build the grading LLM with make_ragas_llm, not a LangChain model. Compare average scores for two answers times three criteria per instruction set, without pass/fail.
    For each answer: Generate, grade the question/answer/reference with three metrics, and display results. Finally show means and save JSON under outputs/evaluations/. Only judge differs from 01_LangChain/03_09_EvalJudge.py; score range, cases, generation, flow, and saving are the same.
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
