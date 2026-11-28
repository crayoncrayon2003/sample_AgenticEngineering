import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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


llm = make_llm("low")

#####################################
# Define the LLM response schema (outline)
#####################################
class Outline(BaseModel):
    # sections: Topics in the order they will be explained
    sections: list[str] = Field(description='Three to five topics in explanation order')


#####################################
# Stage 1: Chain creating an outline
#####################################
prompt_plan = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You write technical explanations. Before writing, plan the topics and their order.'
    ),(
        # user: A message from the application user
        "user",
        'Topic: {topic}'
    ),
])
chain_plan = prompt_plan | llm.with_structured_output(Outline, method="json_schema")

#####################################
# Stage 2: Chain writing according to the outline
#####################################
prompt_write = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You write technical explanations. Write a English explanation of about 600 characters following the supplied outline. '
        'Add a heading for each topic.'
    ),(
        # user: A message from the application user
        "user",
        'Topic: {topic}\n\n[Outline]\n{outline}'
    ),
])
chain_write = prompt_write | llm | StrOutputParser()


def main():
    """Example: Create an outline, then write an explanation following it."""

    topic = 'The importance of reinforcement learning in AI'
    print(f'Topic: {topic}')

    # Stage 1: Create an outline
    result_step1 = chain_plan.invoke({"topic": topic})
    print('--- Stage 1: Outline ---')
    for number, section in enumerate(result_step1.sections, start=1):
        print(f"{number}. {section}")

    # Stage 2: Write according to the outline
    outline = "\n".join(f"{number}. {section}" for number, section in enumerate(result_step1.sections, start=1))
    result_step2 = chain_write.invoke({"topic": topic, "outline": outline})
    print('--- Stage 2: Explanation ---')
    print(result_step2)


if __name__ == "__main__":
    main()
