# 07_Ragas

Examples of Ragas, a library for evaluating RAG and LLM application answers using test cases and metrics.

Ragas evaluation is added to the same examples as 01_LangChain.

Compare files with the same number in 01_LangChain and 07_Ragas to see the differences.

## Setup

```bash
cp 07_Ragas/.env.example 07_Ragas/.env
```

| Variable | Required | Description |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |

Each file disables Ragas usage telemetry by setting `RAGAS_DO_NOT_TRACK=true`.

## LLM used for grading

Ragas metrics require an explicit LLM for grading.

Pass a Ragas-specific LLM created with `llm_factory`, rather than a LangChain model.

02_04_Retrieval.py and 03_09_EvalJudge.py use `make_ragas_llm` to create and pass a Ragas LLM for OpenAI or Claude according to `LLM_PROVIDER`.

This allows switching between OpenAI and Claude with LLM_PROVIDER.

## Notes

- Like 01_LangChain/02_04, `02_04_Retrieval.py` uses embeddings (OpenAI only), so run it with `LLM_PROVIDER=openai`.
- Reasoning models also consume tokens for reasoning, so `max_tokens` is set higher. Too few tokens can interrupt evaluation.

## Differences between DeepEval and Ragas

The two are similar for testing software that includes LLMs, but have different origins and goals.

* DeepEval: Originates from software testing.
* Ragas: Originates from RAG evaluation.

