import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

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
# Define graph state
#####################################
class ResearchState(TypedDict, total=False):
    # request: Request
    # energy / ev / carbon: Three researchers' findings (answers and tools used)
    # report: Combined report
    request: str
    energy: dict
    ev: dict
    carbon: dict
    report: str


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# Stage 1: Three researchers (each node updates a separate field)
def node_energy(state: ResearchState) -> dict:
    return {"energy": research_energy(state["request"])}


def node_ev(state: ResearchState) -> dict:
    return {"ev": research_ev(state["request"])}


def node_carbon(state: ResearchState) -> dict:
    return {"carbon": research_carbon(state["request"])}


# Stage 2: Combine the three research results
def node_merge(state: ResearchState) -> dict:
    report = chain_merge.invoke({
        "energy": state["energy"]["answer"],
        "ev": state["ev"]["answer"],
        "carbon": state["carbon"]["answer"],
    })
    return {"report": report}


#####################################
# Build the graph
#####################################
# START -> energy (renewable-energy researcher)
# ev (electric-vehicle researcher) -> merge (summarizer) -> END
# carbon (carbon-capture researcher)
# Edges from START to three nodes run researchers concurrently. merge waits for all three to finish.
graph = StateGraph(ResearchState)
graph.add_node("energy", node_energy)
graph.add_node("ev", node_ev)
graph.add_node("carbon", node_carbon)
graph.add_node("merge", node_merge)
graph.add_edge(START, "energy")
graph.add_edge(START, "ev")
graph.add_edge(START, "carbon")
graph.add_edge("energy", "merge")
graph.add_edge("ev", "merge")
graph.add_edge("carbon", "merge")
graph.add_edge("merge", END)
app = graph.compile()


def main():
    """Example: Three researchers investigate different fields concurrently; a summarizer combines their findings into one report.
    Roles: One researcher each for renewable energy, electric vehicles, and carbon capture. Each has news and statistics tools and decides which to use and how often. The summarizer uses only their findings.
    Data: Both tools return fictional mock data. Renewable-energy news reports a 40% installation-cost reduction while statistics report 30%. Carbon-capture news exists but statistics do not. Thus identical instructions can yield different tool usage and reports.
    Flow: 1. Three researchers investigate concurrently and summarize in two or three sentences. 2. Display tools used and findings. 3. Combine the findings into one report.
    """

    request = 'Research recent advances in sustainable technologies.'
    print(f'Request: {request}')

    # Run both stages together: Three concurrent researchers, then combine findings.
    result = app.invoke({"request": request})
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
