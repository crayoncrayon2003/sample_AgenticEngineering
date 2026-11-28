import os
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

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

###############################################################################
# Tool 1: Return today's weather for a city, using fictional weather data.
###############################################################################
@tool
def get_weather(city: str) -> str:
    """Return today's weather for the specified city (mock data).

    Args:
        city: City name, e.g., Tokyo, Osaka, Sapporo.
    """
    weather = {
        'Tokyo': 'Sunny, high of 18 degrees Celsius',
        'Osaka': 'Cloudy, high of 17 degrees Celsius',
        'Sapporo': 'Snow, high of 2 degrees Celsius',
    }
    return weather.get(city, f'No weather data are registered for {city}.')

###############################################################################
# Tool 2: Calculate shipping to a destination, using fictional rates.
###############################################################################
@tool
def get_shipping_fee(weight_kg: float, region: Literal['Honshu', 'Hokkaido', 'Okinawa']) -> str:
    """Calculate shipping from package weight and destination region (mock rate table).

    Args:
        weight_kg: Package weight in kg.
        region: Destination region.
    """
    base = {'Honshu': 800, 'Hokkaido': 1200, 'Okinawa': 1500}[region]
    extra = max(0, int(weight_kg) - 2) * 200
    return f'Shipping costs {base + extra} yen ({region}, {weight_kg} kg).'


#####################################
# Pass tools to the LLM
#####################################
tools = [get_weather, get_shipping_fee]
tools_by_name = {"get_weather": get_weather, "get_shipping_fee": get_shipping_fee}
# bind_tools passes the available tool list to the LLM, which returns tool calls as needed.
llm_with_tools = llm.bind_tools(tools)

# system_prompt: The agent's role
system_prompt = (
    'You are a guide. Always use tools to look up weather and shipping fees, and answer using only their results. '
    'If the tools cannot provide the information, say you do not know. '
    'Mention that these are mock data, not real information.'
)


#####################################
# Nodes (graph steps that receive state and return fields to update)
#####################################
# MessagesState holds conversation history (messages). Returned messages are appended to history.

# Call the LLM. Its response is either an answer or a request to call tools.
def node_model(state: MessagesState) -> dict:
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


# Execute tools requested by the LLM.
def node_tools(state: MessagesState) -> dict:
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
    return {"messages": tool_messages}


# Conditional routing: Go to tools if calls exist; otherwise the response is final, so end.
def select_next(state: MessagesState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


#####################################
# Agent (manually connect an LLM node and a tool-execution node in a loop)
#####################################
# START -> model -> END (no tool calls)
#            ↑    │
# Loop through tools when tool calls are present.
graph = StateGraph(MessagesState)
graph.add_node("model", node_model)
graph.add_node("tools", node_tools)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", select_next, ["tools", END])
graph.add_edge("tools", "model")
agent = graph.compile()


def show_tool_calls(messages) -> None:
    """Display tools called by the agent and their results."""
    for message in messages:
        for call in getattr(message, "tool_calls", []):
            print(f"  [Tool call] {call['name']}({call['args']})")
        if message.type == "tool":
            print(f'  [Tool result] {message.text}')

def main():
    """Example: A guide agent that looks up weather and shipping fees using fixed mock data. Five questions cover using one tool, two tools, and no tools."""

    # Five questions
    questions = [
        'Tell me the weather in Osaka.',
        'How much does it cost to ship a 3.5 kg package to Hokkaido?',
        'Tell me the weather in Sapporo and the shipping fee for a 1 kg package to Okinawa.',
        'What is the weather in Naha?',
        'What is an agent that uses tools? Explain briefly.',
    ]
    for question in questions:
        print("----------------------------------------")
        print(f'Question: {question}')
        # Run the agent ("user": A message from the application user).
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result["messages"]

        # Display which tools were called and with which arguments.
        show_tool_calls(messages)
        print(f'Answer: {messages[-1].text}')


if __name__ == "__main__":
    main()
