import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task

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


llm = make_llm("low")  # With minimal reasoning, the model may answer without using tools.


###############################################################################
# Fictional research data: News by field
###############################################################################
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
###############################################################################
# Fictional research data: Statistics by field, with carbon-capture statistics intentionally absent
###############################################################################
STATISTICS = {
    'Renewable energy': "Renewables' share of generation: 20% in 2020 -> 28% in 2025. Solar panel installation costs: down 30% over five years.",
    'Electric vehicles': "Electric vehicles' share of new-car sales: 4% in 2020 -> 15% in 2025.",
}


###############################################################################
# Tool 1: Search news by field
###############################################################################
@tool
def search_news(field: str) -> str:
    """Search recent news for the specified field (fictional mock data).

    Args:
        field: One of "Renewable energy", "Electric vehicles", or "Carbon capture".
    """
    news = NEWS.get(field)
    if not news:
        return f'No news was found for {field}.'
    return "\n".join(f"- {item}" for item in news)


###############################################################################
# Tool 2: Retrieve statistics by field
###############################################################################
@tool
def get_statistics(field: str) -> str:
    """Retrieve statistics for the specified field (fictional mock data).

    Args:
        field: One of "Renewable energy", "Electric vehicles", or "Carbon capture".
    """
    return STATISTICS.get(field, f'No statistics are available for {field}.')


#####################################
# Create researcher agents (one per field).
#####################################
def make_agent_researcher(field: str):
    return create_agent(
        # model: The agent's LLM
        model=llm,
        # tools: Agent tools (the agent decides which to use and how often)
        tools=[search_news, get_statistics],
        # system_prompt: The agent's role
        system_prompt=(
            f"You research '{field}'.\n- Use search_news for news and get_statistics for statistics.\n- If news and statistics conflict, prioritize statistics and mention the discrepancy.\n- If no statistics exist, say the findings could not be verified with statistics.\nSummarize in two or three English sentences and mention that these are fictional mock data."
        ),
    )


agent_energy = make_agent_researcher('Renewable energy')
agent_ev = make_agent_researcher('Electric vehicles')
agent_carbon = make_agent_researcher('Carbon capture')


# Ask the agent. It returns an answer and a list of tools used.
def ask(agent, request: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    tool_calls = [call["name"] for message in result["messages"] if message.type == "ai" for call in message.tool_calls]
    return {"answer": result["messages"][-1].text, "tool_calls": tool_calls}


# Requests for each researcher (separate functions for use with RunnableParallel)
def research_energy(request: str) -> dict:
    return ask(agent_energy, request)


def research_ev(request: str) -> dict:
    return ask(agent_ev, request)


def research_carbon(request: str) -> dict:
    return ask(agent_carbon, request)

#####################################
# Stage 2: Combine the three research results
#####################################
prompt_merge = ChatPromptTemplate.from_messages([
    (
        # system: Instructions from the application developer
        "system",
        'You combine research findings. Write a short English report using only the three supplied research results. '
        'Do not add your own knowledge. Identify which field each finding concerns and state that the data are fictional.'
    ),(
        # user: Findings from the three researchers
        "user",
        '[Renewable energy]\n{energy}\n\n[Electric vehicles]\n{ev}\n\n[Carbon capture]\n{carbon}'
    ),
])
chain_merge = prompt_merge | llm | StrOutputParser()



#####################################
# Tasks (individual workflow steps)
#####################################
@task
def task_energy(request: str) -> dict:
    return research_energy(request)


@task
def task_ev(request: str) -> dict:
    return research_ev(request)


@task
def task_carbon(request: str) -> dict:
    return research_carbon(request)


@task
def merge(energy: str, ev: str, carbon: str) -> str:
    return chain_merge.invoke({"energy": energy, "ev": ev, "carbon": carbon})


#####################################
# Workflow: Three concurrent researchers -> combine findings
#####################################
@entrypoint()
def workflow(request: str) -> dict:
    # Stage 1: Start three tasks before waiting with .result(), allowing researchers to work concurrently.
    future_energy = task_energy(request)
    future_ev = task_ev(request)
    future_carbon = task_carbon(request)
    energy = future_energy.result()
    ev = future_ev.result()
    carbon = future_carbon.result()

    # Stage 2: Combine the three research results
    report = merge(energy["answer"], ev["answer"], carbon["answer"]).result()
    return {"energy": energy, "ev": ev, "carbon": carbon, "report": report}


def main():
    """Example: Three researchers investigate different fields concurrently; a summarizer combines their findings into one report.
    Roles: One researcher each for renewable energy, electric vehicles, and carbon capture. Each has news and statistics tools and decides which to use and how often. The summarizer uses only their findings.
    Data: Both tools return fictional mock data. Renewable-energy news reports a 40% installation-cost reduction while statistics report 30%. Carbon-capture news exists but statistics do not. Thus identical instructions can yield different tool usage and reports.
    Flow: 1. Three researchers investigate concurrently and summarize in two or three sentences. 2. Display tools used and findings. 3. Combine the findings into one report.
    """

    request = 'Research recent advances in sustainable technologies.'
    print(f'Request: {request}')

    # Run both stages together: Three concurrent researchers, then combine findings.
    result = workflow.invoke(request)
    print('--- Stage 1: Findings from three researchers ---')
    print(f"[Renewable energy] Tools used: {result['energy']['tool_calls']}")
    print(result["energy"]["answer"])
    print(f"[Electric vehicles] Tools used: {result['ev']['tool_calls']}")
    print(result["ev"]["answer"])
    print(f"[Carbon capture] Tools used: {result['carbon']['tool_calls']}")
    print(result["carbon"]["answer"])
    print('--- Stage 2: Combined report ---')
    print(result["report"])


if __name__ == "__main__":
    main()
