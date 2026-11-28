# 06_DeepEval

Examples of DeepEval, a library for expressing LLM application evaluations as test cases and metrics.

The evaluation parts of the 01_LangChain examples are replaced with DeepEval.

Compare files with the same number in 01_LangChain and 06_DeepEval to see the differences.

## Setup

```bash
cp 06_DeepEval/.env.example 06_DeepEval/.env
```

| Variable | Required | Description |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |

Each file disables DeepEval usage telemetry by setting `DEEPEVAL_TELEMETRY_OPT_OUT=YES`.

## LLM used for grading

By default, DeepEval calls OpenAI directly for grading.

In 03_09_EvalJudge.py, a LangChain model created by `make_llm` is wrapped in `DeepEvalBaseLLM` and passed to DeepEval.

This allows switching between OpenAI and Claude with LLM_PROVIDER.

## Notes

- `GEval` scores range from 0 to 1 (01_LangChain/03_09 uses 1 to 5).
- `GEval` grading reasons may be in English because DeepEval's grading prompts are in English.

## Differences between DeepEval and Ragas

The two are similar for testing software that includes LLMs, but have different origins and goals.

* DeepEval: Originates from software testing.
* Ragas: Originates from RAG evaluation.

