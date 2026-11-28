# 08_NeMoGuardrails

Examples of NVIDIA NeMo Guardrails, a library that checks LLM inputs and outputs using rails defined in YAML configuration.

The inspection parts of the 01_LangChain examples are replaced with NeMo Guardrails.

Compare files with the same number in 01_LangChain and 08_NeMoGuardrails to see the differences.

## Setup

```bash
cp 08_NeMoGuardrails/.env.example 08_NeMoGuardrails/.env
```

| Variable | Required | Description |
| ---- | :--: | ---- |
| `LLM_PROVIDER` | | LLM provider: `openai` or `anthropic`. Defaults to `openai` |
| `OPENAI_API_KEY` | When using OpenAI | OpenAI API key |
| `OPENAI_MODEL` | | OpenAI model. Defaults to `gpt-5-nano` |
| `ANTHROPIC_API_KEY` | When using Claude | Anthropic API key |
| `ANTHROPIC_MODEL` | | Claude model. Defaults to `claude-haiku-5-5` |

Each file disables NeMo Guardrails usage telemetry by setting `NEMO_GUARDRAILS_NO_USAGE_STATS=1`.

## LLM used for input screening

NeMo Guardrails normally specifies the screening LLM in `models` in its YAML configuration.

03_01_InputCheck.py and 03_02_OutputCheck.py instead pass a LangChain model created by `make_llm` to `LLMRails`.

This allows switching between OpenAI and Claude with LLM_PROVIDER.

The `regex check output` inspection in 03_02_OutputCheck.py uses regular expressions, so it does not use an LLM.

## Why there is no 03_04 (tool argument validation)

NeMo Guardrails has `tool_call` rails for inspecting tool calls, but the same example as 01_LangChain/03_04 cannot be written for these reasons.

- It runs only on the new execution engine (IORails), not on `LLMRails` used in this folder.
- An additional CEL expression-language package is required for validation expressions.
- The validation cannot use runtime information needed to compare the requested user ID with the logged-in user's ID, the key point of 03_04.

## Notes

- `self check input` returns only an allow/reject answer (Yes/No), not the violated rules or reasons.
- `regex check output` does not identify which pattern matched.
- Rail configuration is in each file's `RAILS_CONFIG` (YAML). It can also be written in `config.yml` and loaded with `RailsConfig.from_path`.

## Differences between NeMo Guardrails and Guardrails AI

The two are similar for checking LLM inputs and outputs, but have different origins and goals.

* NeMo Guardrails: Originates from conversation control. YAML-configured rails govern input, output, and dialogue flow.
* Guardrails AI: Originates from output validation. Combines validators to check output structure and content.
