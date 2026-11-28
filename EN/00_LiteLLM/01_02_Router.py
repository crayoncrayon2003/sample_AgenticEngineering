import os
from typing import Literal

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
# Define the LLM response schema
#####################################
class Route(BaseModel):
    # reason: Why this handler was selected.
    # destination: The selected handler; one of "expense", "it", or "unclear".
    # Order the fields as reason, then destination.
    # The LLM outputs fields in order. Asking for a reason first helps it select a handler consistent with that reason.
    reason: str = Field(description='The inquiry and the reasoning for the appropriate handler (one sentence in English)')
    destination: Literal["expense", "it", "unclear"] = Field(description='expense: Expense reimbursement; it: PC or network issues; unclear: Cannot determine')


prompt_route = [
    # system: Instructions from the application developer
    {"role": "system", "content": 'You are the receptionist for an internal help desk. Read the inquiry and select one handler. If you cannot identify what the inquiry is about, select unclear.'},
    # user: A message from the application user
    {"role": "user", "content": "{request}"},
]


# Equivalent to LangChain's chain_route.invoke({...}). Accept prompt values as a dictionary.
def chain_route(values: dict) -> Route:
    # Passing a Pydantic class as response_format returns JSON matching that schema.
    message = call_llm(llm, format_prompt(prompt_route, values), response_format=Route)
    return Route.model_validate_json(message.content)


#####################################
# Processing for each handler
#####################################
def make_chain_handler(role: str):
    prompt_handler = [
        # system: Instructions from the application developer
        {"role": "system", "content": role + 'Answer in English in no more than three sentences.'},
        # user: A message from the application user
        {"role": "user", "content": "{request}"},
    ]

    # Equivalent to LangChain's chain_handler.invoke({...}). Accept prompt values as a dictionary.
    def chain_handler(values: dict) -> str:
        return call_llm(llm, format_prompt(prompt_handler, values)).content

    return chain_handler


chain_handlers = {
    "expense": make_chain_handler('You handle accounting. Explain the expense reimbursement procedure. '
                                  'For anything unclear in the internal rules, recommend checking with the accounting department.'),
    "it"     : make_chain_handler('You handle IT. Explain checks the user can perform.'),
    "unclear": make_chain_handler('You are the receptionist. Since the request is unclear, '
                                  'ask specifically what the user needs help with.'),
}


def main():
    """Example: Route three internal help desk inquiries to expense, IT, or unclear."""

    requests = [
        'I want to claim a taxi fare from a business trip, but I lost the receipt.',
        "I cannot connect to the meeting room's Wi-Fi.",
        'What is happening with that matter?',
    ]

    for request in requests:
        print("----------------------------------------")
        print(f'Inquiry: {request}')

        # Stage 1: Select a handler
        result_step1 = chain_route({"request": request})
        print(f'Selected handler: {result_step1.destination}')
        print(f'Routing reason: {result_step1.reason}')

        # Stage 2: Answer using the selected handler
        result_step2 = chain_handlers[result_step1.destination]({"request": request})

        print(f'Answer: {result_step2}')


if __name__ == "__main__":
    main()
