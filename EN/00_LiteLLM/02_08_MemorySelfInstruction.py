import json
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


llm = make_llm("minimal")

#####################################
# Instruction storage
#####################################
# Save the initial instructions.
instruction_store = {"instructions": 'You explain technical terms. Explain in English.'}


# Read the saved instructions.
def load_instructions() -> str:
    return instruction_store["instructions"]


#####################################
# Revise instructions
#####################################
# Define the LLM response schema (instruction revision)
class InstructionUpdate(BaseModel):
    # request: The user's preferences for future answers.
    # changed: True if the instructions should be rewritten.
    # instructions: The entire revised instructions.
    # Order request before changed so the model states the preference before deciding.
    request: str = Field(description="The user's preference for future answers; an empty string if none")
    changed: bool = Field(description='true if request is nonempty and the instructions should change')
    instructions: str = Field(description='The entire updated instructions. Return the current instructions unchanged if no update is needed.')


prompt_revise = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        "You manage an agent's instructions (system prompt).\n"
        '- Only if the user clearly requests a change to future answers (length, format, tone, etc.), '
        'set changed to true. Set it to false for ordinary questions alone.\n'
        '- When updating, add the request to the current instructions and return the entire instructions '
        "as directions to the agent starting with 'You are ...'. "
        "Do not include conversation history or explanations such as 'in this case'."
    )},
    # user: A message from the application user
    {"role": "user", "content": '[Current instructions]\n{current}\n\n[Conversation]\n{conversation}'},
]
# Use a model with slightly deeper reasoning because detecting preferences requires difficult judgments.
llm_revise = make_llm("low")


# Equivalent to LangChain's chain_revise.invoke({...}). Accept prompt values as a dictionary.
def chain_revise(values: dict) -> InstructionUpdate:
    response = call_llm(llm_revise, format_prompt(prompt_revise, values), response_format=InstructionUpdate)
    return InstructionUpdate.model_validate_json(response.content)


#####################################
# Agent: Answer using saved instructions -> revise them if preferences are expressed
#####################################
# History for each conversation
threads = {}


def run_agent(message: str, thread_id: str) -> str:
    history = threads.get(thread_id, [])
    history.append({"role": "user", "content": message})

    # Step 1: Answer using saved instructions (dynamic_prompt middleware in the LangChain example).
    answer = call_llm(llm, [{"role": "system", "content": load_instructions()}] + history).content
    history.append({"role": "assistant", "content": answer})
    threads[thread_id] = history

    # Step 2: Revise instructions if preferences are expressed (after_agent middleware in the LangChain example).
    result_revise = chain_revise({
        "current": load_instructions(),
        "conversation": json.dumps(history, ensure_ascii=False),
    })
    if result_revise.changed:
        instruction_store["instructions"] = result_revise.instructions
        print(f'  [Instructions updated] {result_revise.instructions}')
    else:
        print('  [Instructions unchanged]')
    return answer


def main():
    """Example: An AI assistant remembers response preferences as its own instructions and applies them across conversations. It explains technical terms and stores response instructions. Tanaka asks about terms and states preferred response styles."""

    print(f'Initial instructions: {load_instructions()}')

    # Three messages (conversation ID, message)
    # Conversation 1 (thread-1): Tanaka asks the assistant what an API is.
    # The assistant answers according to the initial instructions.
    # Tanaka requests explanations within three sentences, with one familiar analogy.
    # The assistant saves instructions incorporating the request.
    # Conversation 2 (thread-2): In a different conversation, Tanaka asks what a database is.
    # Following the instructions saved in conversation 1, the assistant answers within three sentences with an analogy.
    messages = [
        ("thread-1", 'What is an API?'),
        ("thread-1", 'That is too long. From now on, explain within three sentences and include one familiar analogy.'),
        ("thread-2", 'What is a database?'),  # The "thread-id" is different.
    ]
    for thread_id, message in messages:
        print("----------------------------------------")
        print(f'[{thread_id}] User: {message}')
        # Run the agent
        answer = run_agent(message, thread_id)
        print(f"[{thread_id}] AI：{answer}")

    print(f'Final instructions: {load_instructions()}')


if __name__ == "__main__":
    main()
