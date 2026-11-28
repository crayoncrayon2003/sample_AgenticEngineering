import json
import os
from concurrent.futures import ThreadPoolExecutor

import litellm
from dotenv import load_dotenv

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


llm = make_llm("low")  # With minimal reasoning, the model may answer without using tools.

#####################################
# Fictional research data
#####################################
# Fictional news by field
NEWS = {
    'Renewable energy': [
        'An offshore wind farm began operating in one region and is expected to supply electricity for 100,000 households.',
        'A report claimed solar panel installation costs had fallen 40% from five years ago.',
    ],
    'Electric vehicles': [
        'A company prototyped a battery that charges to 80% in 10 minutes. Mass production is undecided.',
    ],
    'Carbon capture': [
        'Estimated costs of equipment capturing atmospheric CO2 fell 30%. Demonstrations remain small-scale.',
    ],
}
# Fictional statistics by field. No statistics are available for carbon capture.
STATISTICS = {
    'Renewable energy': "Renewables' share of generation: 20% in 2020 -> 28% in 2025. Solar panel installation costs: down 30% over five years.",
    'Electric vehicles': "Electric vehicles' share of new-car sales: 4% in 2020 -> 15% in 2025.",
}


#####################################
# Tool 1: Search news by field
#####################################
def search_news(field: str) -> str:
    """Search recent news for the specified field (fictional mock data).

    Args:
        field: One of "Renewable energy", "Electric vehicles", or "Carbon capture".
    """
    news = NEWS.get(field)
    if not news:
        return f'No news was found for {field}.'
    return "\n".join(f"- {item}" for item in news)


#####################################
# Tool 2: Retrieve statistics by field
#####################################
def get_statistics(field: str) -> str:
    """Retrieve statistics for the specified field (fictional mock data).

    Args:
        field: One of "Renewable energy", "Electric vehicles", or "Carbon capture".
    """
    return STATISTICS.get(field, f'No statistics are available for {field}.')


#####################################
# Tool descriptions sent to the LLM
#####################################
# LangChain's @tool generates this description from the docstring and type hints. Here we write a dictionary manually.
tools = [
    {"type": "function", "function": {
        "name": "search_news",
        "description": 'Search recent news for the specified field (fictional mock data).',
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string", "description": 'Field: "Renewable energy", "Electric vehicles", or "Carbon capture"'}},
            "required": ["field"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_statistics",
        "description": 'Retrieve statistics for the specified field (fictional mock data).',
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string", "description": 'Field: "Renewable energy", "Electric vehicles", or "Carbon capture"'}},
            "required": ["field"],
        },
    }},
]
tool_functions = {"search_news": search_news, "get_statistics": get_statistics}


#####################################
# Researcher (one per field)
#####################################
# Create a researcher's system_prompt (the agent decides which tools to call and how often).
def make_system_prompt(field: str) -> str:
    return (
        f"You research '{field}'.\n- Use search_news for news and get_statistics for statistics.\n- If news and statistics conflict, prioritize statistics and mention the discrepancy.\n- If no statistics exist, say the findings could not be verified with statistics.\nSummarize in two or three English sentences and mention that these are fictional mock data."
    )


# Ask the researcher and return its answer and tool list (LLM -> tool execution -> LLM loop).
def ask(field: str, request: str) -> dict:
    messages = [
        {"role": "system", "content": make_system_prompt(field)},
        {"role": "user", "content": request},
    ]
    tool_calls = []
    response = call_llm(llm, messages, tools=tools)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_calls.append(tool_call.function.name)
            result = tool_functions[tool_call.function.name](**json.loads(tool_call.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        response = call_llm(llm, messages, tools=tools)
    return {"answer": response.content, "tool_calls": tool_calls}



#####################################
# Stage 1: Three researchers investigate concurrently
#####################################
# ThreadPoolExecutor runs work concurrently in separate threads (equivalent to LangChain's RunnableParallel).
def chain_parallel(request: str) -> dict:
    with ThreadPoolExecutor() as executor:
        future_energy = executor.submit(ask, 'Renewable energy', request)
        future_ev = executor.submit(ask, 'Electric vehicles', request)
        future_carbon = executor.submit(ask, 'Carbon capture', request)
        return {"energy": future_energy.result(), "ev": future_ev.result(), "carbon": future_carbon.result()}


#####################################
# Stage 2: Combine the three research results
#####################################
prompt_merge = [
    # system: Instructions from the application developer
    {"role": "system", "content": (
        'You combine research findings. Write a short English report using only the three supplied research results. '
        'Do not add your own knowledge. Identify which field each finding concerns and state that the data are fictional.'
    )},
    # user: A message from the application user
    {"role": "user", "content": (
        '[Renewable energy]\n{energy}\n\n[Electric vehicles]\n{ev}\n\n[Carbon capture]\n{carbon}'
    )},
]

# Equivalent to LangChain's chain_merge.invoke({...}). Accept prompt values as a dictionary.
def chain_merge(values: dict) -> str:
    message = call_llm(llm, format_prompt(prompt_merge, values))
    return message.content


def main():
    """Example: Three researchers investigate different fields concurrently; a summarizer combines their findings into one report.
    Roles: One researcher each for renewable energy, electric vehicles, and carbon capture. Each has news and statistics tools and decides which to use and how often. The summarizer uses only their findings.
    Data: Both tools return fictional mock data. Renewable-energy news reports a 40% installation-cost reduction while statistics report 30%. Carbon-capture news exists but statistics do not. Thus identical instructions can yield different tool usage and reports.
    Flow: 1. Three researchers investigate concurrently and summarize in two or three sentences. 2. Display tools used and findings. 3. Combine the findings into one report.
    """

    request = 'Research recent advances in sustainable technologies.'
    print(f'Request: {request}')

    # Stage 1: Three researchers investigate concurrently
    result_step1 = chain_parallel(request)
    print('--- Stage 1: Findings from three researchers ---')
    print(f"[Renewable energy] Tools used: {result_step1['energy']['tool_calls']}")
    print(result_step1["energy"]["answer"])
    print(f"[Electric vehicles] Tools used: {result_step1['ev']['tool_calls']}")
    print(result_step1["ev"]["answer"])
    print(f"[Carbon capture] Tools used: {result_step1['carbon']['tool_calls']}")
    print(result_step1["carbon"]["answer"])

    # Stage 2: The summarizer creates one report
    result_step2 = chain_merge({
        "energy": result_step1["energy"]["answer"],
        "ev": result_step1["ev"]["answer"],
        "carbon": result_step1["carbon"]["answer"],
    })
    print('--- Stage 2: Combined report ---')
    print(result_step2)


if __name__ == "__main__":
    main()
