import json
import os

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


llm = make_llm("minimal")


###############################################################################
# Tool 1: Return today's weather for a city, using fictional weather data.
###############################################################################
def get_weather(city: str) -> str:
    """Return today's weather for the specified city (mock data)."""
    weather = {
        'Tokyo': 'Sunny, high of 18 degrees Celsius',
        'Osaka': 'Cloudy, high of 17 degrees Celsius',
        'Sapporo': 'Snow, high of 2 degrees Celsius',
    }
    return weather.get(city, f'No weather data are registered for {city}.')

###############################################################################
# Tool 2: Calculate shipping to a destination, using fictional rates.
###############################################################################
def get_shipping_fee(weight_kg: float, region: str) -> str:
    """Calculate shipping from package weight and destination region (mock rate table)."""
    base = {'Honshu': 800, 'Hokkaido': 1200, 'Okinawa': 1500}[region]
    extra = max(0, int(weight_kg) - 2) * 200
    return f'Shipping costs {base + extra} yen ({region}, {weight_kg} kg).'


#####################################
# Tool descriptions sent to the LLM
#####################################
# LangChain's @tool generates these descriptions from function docstrings and type hints. Here we write them manually.
tools = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Return today's weather for the specified city (mock data).",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": 'City name, e.g., Tokyo, Osaka, or Sapporo'}},
            "required": ["city"],
        },
    }},
    {"type": "function", "function": {
        "name": "get_shipping_fee",
        "description": 'Calculate shipping from package weight and destination region (mock rate table).',
        "parameters": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": 'Package weight in kg'},
                "region": {"type": "string", "enum": ['Honshu', 'Hokkaido', 'Okinawa'], "description": 'Destination region'},
            },
            "required": ["weight_kg", "region"],
        },
    }},
]
# A lookup table mapping tool names to functions
tool_functions = {"get_weather": get_weather, "get_shipping_fee": get_shipping_fee}


#####################################
# Agent (manually implement the loop that LangChain's create_agent runs internally)
#####################################
# system_prompt: The agent's role
system_prompt = (
    'You are a guide. Always use tools to look up weather and shipping fees, and answer using only their results. '
    'If the tools cannot provide the information, say you do not know. '
    'Mention that these are mock data, not real information.'
)


def run_agent(question: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    message = call_llm(llm, messages, tools=tools)

    # While the LLM requests tool calls, execute the tools and call the LLM again.
    while message.tool_calls:
        messages.append(message)
        for tool_call in message.tool_calls:
            # Tool arguments are returned as a JSON string.
            args = json.loads(tool_call.function.arguments)
            result = tool_functions[tool_call.function.name](**args)
            print(f'  [Tool call] {tool_call.function.name}({args})')
            print(f'  [Tool result] {result}')
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        message = call_llm(llm, messages, tools=tools)

    # Once there are no tool calls, the response is the final answer.
    messages.append(message)
    return messages


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
        # Run the agent (tool calls and results are displayed during execution).
        messages = run_agent(question)
        print(f'Answer: {messages[-1].content}')


if __name__ == "__main__":
    main()
