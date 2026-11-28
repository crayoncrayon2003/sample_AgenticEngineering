import os

from dotenv import load_dotenv
from langchain.agents import create_agent
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


llm = make_llm("minimal")

#####################################
# Agent 1: Researcher
#####################################
agent_researcher = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: Agent tools (none here)
    tools=[],
    # system_prompt: The agent's role
    system_prompt=(
        'You research and organize AI topics. Summarize major trends, practical uses, impacts, and caveats in English. '
        'No search tool is available; answer within your existing knowledge. '
        'Clearly identify uncertainty and do not invent unverified sources or URLs.'
    ),
)

#####################################
# Agent 2: Writer
#####################################
agent_writer = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: Agent tools (none here)
    tools=[],
    # system_prompt: The agent's role
    system_prompt=(
        'You write technical articles. Use the supplied research summary to write a English blog post for general readers. '
        'Include a title and headings and use about 200 characters. '
        'Do not assert uncertain findings as facts or add unsupported numbers or sources.'
    ),
)

#####################################
# Define graph state (data exchanged between the two agents)
#####################################
class TeamState(TypedDict, total=False):
    # topic: Topic
    # findings: Researcher's findings
    # article: Writer's article
    topic: str
    findings: str
    article: str


#####################################
# Nodes: Ask each agent to perform its work
#####################################
# Stage 1: The researcher organizes the topic.
def node_research(state: TeamState) -> dict:
    topic = state["topic"]
    return {"findings": agent_researcher.invoke({"messages": [{"role": "user", "content": f"Topic: {topic}\nSelect three major trends and describe each trend's key points, practical uses, impacts, and caveats."}]})["messages"][-1].text}


# Stage 2: The writer receives the findings and writes an article
def node_write(state: TeamState) -> dict:
    topic = state["topic"]
    findings = state["findings"]
    return {"article": agent_writer.invoke({"messages": [{"role": "user", "content": f'Topic: {topic}\nWrite a blog post based on these findings.\n\n[Findings]\n{findings}'}]})["messages"][-1].text}


#####################################
# Build the graph
#####################################
# START -> research (researcher) -> write (writer) -> END
graph = StateGraph(TeamState)
graph.add_node("research", node_research)
graph.add_node("write", node_write)
graph.add_edge(START, "research")
graph.add_edge("research", "write")
graph.add_edge("write", END)
app = graph.compile()


def main():
    """Example: Two agents hand off work in sequence. The researcher organizes major trends; the writer turns those findings into a blog post."""

    topic = 'Major trends in AI agents'
    print(f'Topic: {topic}')

    # Run both stages together: Researcher organizes findings, then writer creates an article.
    result = app.invoke({"topic": topic})
    print('--- Stage 1: Research findings ---')
    print(result["findings"])
    print("--- Stage 2: Writer's article ---")
    print(result["article"])


if __name__ == "__main__":
    main()
