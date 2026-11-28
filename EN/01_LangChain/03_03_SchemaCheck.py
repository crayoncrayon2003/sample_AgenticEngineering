import json
import os

from dotenv import load_dotenv
from jsonschema import Draft202012Validator, FormatChecker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
SCHEMA_PATH = os.path.join(ROOT, "data", "schemas", "user_profile.schema.json")


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


llm = make_llm("minimal")

#####################################
# Load the JSON Schema file defining output structure and constraints.
#####################################
with open(SCHEMA_PATH, encoding="utf-8") as file:
    schema = json.load(file)
# Check whether the schema itself is valid.
Draft202012Validator.check_schema(schema)


#####################################
# Stage 1: Chain extracting user information from text as JSON
#####################################
prompt_extract = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You extract user information from text. '
        'Return only a JSON object matching the following JSON Schema. Do not add explanations or code fences. '
        'Do not guess or add values absent from the text. '
        'Copy email addresses and birth dates exactly as written, even if their formats are invalid. '
        'Python will validate them later.\n\n'
        "JSON Schema：\n{schema}"
    ),(
        # user: A message from the application user
        "user",
        'Text to extract from:\n{text}'
    ),
])
chain_extract = prompt_extract | llm | StrOutputParser()


#####################################
# Stage 2: Validate returned JSON against JSON Schema
#####################################
def validate_output(raw_json: str) -> dict:
    # Can it be parsed as JSON?
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as error:
        return {"status": "invalid_json", "errors": [str(error)]}

    # Does it meet schema constraints (required fields, types, email and date formats, etc.)?
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = [f"{error.json_path}：{error.message}" for error in validator.iter_errors(data)]
    if errors:
        return {"status": "schema_mismatch", "errors": errors}
    return {"status": "valid", "errors": []}


def main():
    """Example: Validate AI-extracted JSON against a predefined JSON Schema. Of two texts, the second has invalid email and birth-date formats."""

    print(f'Schema: {SCHEMA_PATH}')

    # Two texts
    texts = [
        'My name is Hanako Yamada. My email is hanako@example.com. My birth date is 1995-07-21. My interests are Python and data analysis.',
        'My name is Taro Yamada. My recorded email is taro@example.com. My recorded birth date is 1995-02-30. My interest is machine learning.',
    ]
    for text in texts:
        print("----------------------------------------")
        print(f'Text: {text}')

        # Stage 1: Extract user information from text as JSON
        result_step1 = chain_extract.invoke({"schema": json.dumps(schema, ensure_ascii=False), "text": text})
        print(f'JSON returned by the AI: {result_step1}')

        # Stage 2: Validate against JSON Schema
        result_step2 = validate_output(result_step1)
        print(f"Validation result: {result_step2['status']}")
        for error in result_step2["errors"]:
            print(f"  - {error}")
        if result_step2["status"] == "valid":
            print('The data matches the schema and can be passed to subsequent processing.')
        else:
            print('Validation failed. Do not pass the data to subsequent processing.')


if __name__ == "__main__":
    main()
