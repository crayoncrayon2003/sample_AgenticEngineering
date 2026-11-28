import os

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic


def make_llm(effort: str) -> dict:
    """Create LiteLLM model settings based on LLM_PROVIDER. LiteLLM selects the LLM using a 'provider/model' string. effort controls reasoning depth; Claude has no minimal level, so use low instead."""
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"Set OPENAI_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"Set ANTHROPIC_API_KEY in {os.path.join(ROOT, '.env')}.")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError('Set LLM_PROVIDER to openai or anthropic.')


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """Call the LLM through LiteLLM and return its message (equivalent to LangChain's llm.invoke)."""
    response = litellm.completion(
        model=llm["model"],
        reasoning_effort=llm["reasoning_effort"],
        max_tokens=llm["max_tokens"],
        messages=messages,
        response_format=response_format,
        tools=tools,
        timeout=60,
        num_retries=1,
    )
    return response.choices[0].message


def format_prompt(prompt: list, values: dict) -> list:
    """Fill named placeholders in the prompt with values (equivalent to LangChain's ChatPromptTemplate)."""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("low")

#####################################
# Define the LLM response schema (outline)
#####################################
class Outline(BaseModel):
    # sections: Topics in the order they will be explained
    sections: list[str] = Field(description='Three to five topics in explanation order')


#####################################
# Stage 1: Create an outline (plan)
#####################################
prompt_plan = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You write technical explanations. Before writing, plan the topics and their order.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        'Topic: {topic}'
    )},
]

# Equivalent to LangChain's chain_plan.invoke({...}). Accept prompt values as a dictionary.
def chain_plan(values: dict) -> Outline:
    message = call_llm(llm, format_prompt(prompt_plan, values), response_format=Outline)
    return Outline.model_validate_json(message.content)


#####################################
# Stage 2: Write according to the outline
#####################################
prompt_write = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You write technical explanations. Write a English explanation of about 600 characters following the supplied outline. '
        'Add a heading for each topic.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        'Topic: {topic}\n\n[Outline]\n{outline}'
    )},
]

# Equivalent to LangChain's chain_write.invoke({...}). Accept prompt values as a dictionary.
def chain_write(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_write, values))
    return message.content


def main():
    """Example: Create an outline, then write an explanation following it."""

    topic = 'The importance of reinforcement learning in AI'
    print(f'Topic: {topic}')

    # Stage 1: Create an outline
    result_step1 = chain_plan({"topic": topic})
    print('--- Stage 1: Outline ---')
    for number, section in enumerate(result_step1.sections, start=1):
        print(f"{number}. {section}")

    # Stage 2: Write according to the outline
    outline = "\n".join(f"{number}. {section}" for number, section in enumerate(result_step1.sections, start=1))
    result_step2 = chain_write({"topic": topic, "outline": outline})
    print('--- Stage 2: Explanation ---')
    print(result_step2)


if __name__ == "__main__":
    main()
