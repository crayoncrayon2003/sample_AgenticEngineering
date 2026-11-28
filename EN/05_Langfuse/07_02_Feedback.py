import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

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


#####################################
# Langfuse: Destination for execution records (traces)
#####################################
# Omitting the logging destination sends data to Langfuse Cloud. Send it to local Langfuse instead.
os.environ.setdefault("LANGFUSE_BASE_URL", "http://localhost:3000")
langfuse = get_client()

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
# Agent
#####################################
agent = create_agent(
    # model: The agent's LLM
    model=llm,
    # tools: The agent's tools
    tools=[get_weather, get_shipping_fee],
    # system_prompt: The agent's role
    system_prompt=(
        'You are a guide. Always use tools to look up weather and shipping fees, and answer using only their results. '
        'If the tools cannot provide the information, say you do not know. '
        'Mention that these are mock data, not real information.'
    ),
)


#####################################
# Automatic evaluation: Check whether the answer follows predefined rules
#####################################
# Automatically score programmatically checkable conditions on every run without waiting for user feedback.
def check_rule(answer: str) -> bool:
    """Does the answer mention that the data are mock data?"""
    return 'mock' in answer.lower()


def main():
    """Example: Score answers from the weather/shipping guide agent (same as 01_LangChain/02_01_ToolFunction.py) and send scores to Langfuse.
    Scores: 1. user_feedback simulates helpful/unhelpful buttons, using predefined values here (real apps get them from the UI). 2. mentions_mock_data programmatically checks whether the answer identifies mock data.
    Remember the generating trace_id and attach scores later with create_score, as user feedback arrives after the answer. The trace list gains score columns; Scores shows summaries such as helpfulness rates. Filter low-scoring traces to investigate failed questions.
    """

    # Three questions with user ratings (True: helpful; False: unhelpful) and comments
    conversations = [
        ('Tell me the weather in Osaka.', True, 'It was convenient to get the answer quickly.'),
        ('What is the weather in Naha?', False, "I wanted Naha's weather but could not get it."),
        ('How much does it cost to ship a 3.5 kg package to Hokkaido?', True, ""),
    ]
    for question, helpful, comment in conversations:
        print("----------------------------------------")
        print(f'Question: {question}')

        # 1. Generate an answer and remember its trace ID.
        with langfuse.start_as_current_observation(name='Guide agent', input=question) as span:
            with propagate_attributes(trace_name='Guide agent', tags=["07_02_Feedback"]):
                result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config={"callbacks": [CallbackHandler()]})
            answer = result["messages"][-1].text
            span.update(output=answer)
            trace_id = langfuse.get_current_trace_id()
        print(f'Answer: {answer}')

        # 2. Add automatic evaluation.
        rule_ok = check_rule(answer)
        langfuse.create_score(trace_id=trace_id, name="mentions_mock_data", value=1 if rule_ok else 0, data_type="BOOLEAN",
                              comment='Mentions that the data are mock data' if rule_ok else 'Does not mention that the data are mock data')
        print(f"Automatic evaluation (mock data): {'Mentioned' if rule_ok else 'Not mentioned'}")

        # 3. Add user feedback (assumed to arrive after returning the answer).
        langfuse.create_score(trace_id=trace_id, name="user_feedback", value=1 if helpful else 0, data_type="BOOLEAN", comment=comment)
        print(f"User feedback: {'Helpful' if helpful else 'Unhelpful'} ({comment or 'No comment'})")
        print(f'Trace: {langfuse.get_trace_url(trace_id=trace_id)}')

    # Records are batched in the background. Flush all remaining records before program exit.
    langfuse.flush()


if __name__ == "__main__":
    main()
