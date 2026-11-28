import os
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

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
# Tasks (individual workflow steps)
#####################################
# Call the LLM. Its response is either an answer or a request to call tools.
@task
def call_model(messages: list) -> AIMessage:
    return llm_with_tools.invoke([SystemMessage(content=system_prompt)] + messages)


# Execute one tool requested by the LLM.
@task
def call_tool(tool_call: dict) -> ToolMessage:
    return tools_by_name[tool_call["name"]].invoke(tool_call)


#####################################
# Agent
#####################################
@entrypoint()
def agent(question: str) -> list:
    messages = [HumanMessage(content=question)]
    response = call_model(messages).result()

    # While the LLM requests tool calls, execute the tools and call the LLM again.
    while response.tool_calls:
        # Execute requested tools concurrently and wait for all results.
        futures = [call_tool(tool_call) for tool_call in response.tool_calls]
        tool_messages = [future.result() for future in futures]

        # Append the LLM response and tool results to conversation history.
        messages = add_messages(messages, [response] + tool_messages)
        response = call_model(messages).result()

    # Once there are no tool calls, the response is the final answer.
    return add_messages(messages, [response])


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
        messages = agent.invoke(question)

        # Display which tools were called and with which arguments.
        show_tool_calls(messages)
        print(f'Answer: {messages[-1].text}')


if __name__ == "__main__":
    main()
