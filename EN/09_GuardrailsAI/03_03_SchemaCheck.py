import datetime
import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from guardrails import Guard
from guardrails.settings import settings
from guardrails.validators import FailResult, PassResult, Validator, register_validator
from pydantic import BaseModel, ConfigDict, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


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
# Prepare Guardrails AI
#####################################
# Disable telemetry, which otherwise defaults to enabled when ~/.guardrailsrc is absent.
# Validators use settings from creation time, so disable telemetry before creating them; also disable it through Guard.configure.
settings.rc.enable_metrics = False


#####################################
# Define output structure and constraints using a Pydantic class.
#####################################
# Express the same constraints as 01_LangChain/03_03_SchemaCheck.py's JSON Schema (data/schemas/user_profile.schema.json) in Pydantic.
class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Reject unspecified fields (additionalProperties: false).
    name: str = Field(min_length=1, description="User's full name.")
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="User's email address.")
    date_of_birth: Optional[str] = Field(default=None, description='Birth date; YYYY-MM-DD when supplied as a string.')
    interests: list[str] = Field(default_factory=list, description='List of fields of interest.')


# Schema sent to the LLM (JSON Schema), generated from the Pydantic class
schema = UserProfile.model_json_schema()


# Validator: Use Python to check constraints not covered by types or formats.
# Reject nonexistent dates such as 1995-02-30 even if their format is correct.
@register_validator(name="sample/existing-date", data_type="string")
class ExistingDate(Validator):
    def _validate(self, value, metadata):
        if value is None:
            return PassResult()
        try:
            datetime.date.fromisoformat(value)
        except ValueError:
            return FailResult(error_message=f'Not an existing date in YYYY-MM-DD format: {value}')
        return PassResult()


# Guard.for_pydantic creates a Guard validating LLM output against the Pydantic class.
# use(..., on="$.date_of_birth"): Add a validator only to the specified field.
guard = Guard.for_pydantic(UserProfile).use(ExistingDate(on_fail="noop"), on="$.date_of_birth")
guard.configure(allow_metrics_collection=False)


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
# Stage 2: Validate returned JSON with Guard
#####################################
def validate_output(raw_json: str) -> dict:
    # parse loads LLM output and checks Pydantic constraints and validators. num_reasks=0 disables LLM regeneration requests.
    outcome = guard.parse(raw_json, num_reasks=0)

    # Can it be parsed as JSON?
    if outcome.error:
        return {"status": "invalid_json", "errors": [outcome.error]}

    # Check class constraints (required fields, types, email format, etc.) and the validator (whether the date exists).
    errors = [f"{summary.property_path}：{summary.failure_reason}" for summary in outcome.validation_summaries or []]
    if outcome.reask is not None:
        errors += [result.error_message for result in outcome.reask.fail_results]
    if not outcome.validation_passed:
        return {"status": "schema_mismatch", "errors": errors}
    return {"status": "valid", "errors": []}

def main():
    """Example: Validate AI-extracted user JSON with Guardrails AI Guard and Pydantic UserProfile; do not forward invalid data. UserProfile requires nonempty names, email format, no unspecified fields, etc.; ExistingDate checks date_of_birth is an existing YYYY-MM-DD date. Hanako Yamada's valid record passes; Taro Yamada's nonexistent 1995-02-30 fails.
    guard.parse validates without an LLM: invalid_json if unparseable, schema_mismatch with failed constraints if invalid, valid if all checks pass. UserProfile's format checks alone cannot detect 1995-02-30, so use a validator.
    Flow: Extract user JSON, validate with Guard, display results, and allow subsequent processing only on success. Unlike 01_LangChain/03_03_SchemaCheck.py's JSON Schema file and jsonschema, use a Pydantic class and Guard plus a custom date validator. Texts, extraction, and flow are the same.
    """

    print(f'Schema: {UserProfile.__name__}')

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
