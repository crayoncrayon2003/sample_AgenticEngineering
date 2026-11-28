import os

import litellm
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TaskState, TextPart
from a2a.utils import new_task
from dotenv import load_dotenv

# --- Configuration ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai or anthropic
HOST = "127.0.0.1"
PORT = 9000
SERVER_URL = f"http://{HOST}:{PORT}/"


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
# Published agent: Technical explainer
#####################################
# system_prompt: The agent's role
system_prompt = 'You explain technology. Explain the requested topic in beginner-friendly English within three sentences.'


# Ask the technical explainer and return the answer (use litellm.acompletion because the A2A server is asynchronous).
async def agent_expert(message: str) -> str:
    response = await litellm.acompletion(
        model=llm["model"], reasoning_effort=llm["reasoning_effort"], max_tokens=llm["max_tokens"],
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
        timeout=60, num_retries=1,
    )
    return response.choices[0].message.content


#####################################
# Forward A2A requests to the agent
#####################################
class ExpertAgentExecutor(AgentExecutor):
    # Called when an A2A request arrives
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Register the request as a task and mark it as working.
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.working)

        # Run the agent
        answer = await agent_expert(context.get_user_input())

        # Register the answer as an artifact and mark the task as completed.
        await updater.add_artifact(parts=[Part(root=TextPart(text=answer))], name="answer")
        await updater.complete()

    # Request cancellation is unsupported.
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError('Cancellation is unsupported.')


#####################################
# Agent card: A public description of the agent's name, capabilities, and endpoint
#####################################
agent_card = AgentCard(
    name="Technical Explanation Agent",
    description='An agent that explains technical questions in beginner-friendly English.',
    url=SERVER_URL,
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[AgentSkill(
        id="explain_technical_topic",
        name='Technical explanation',
        description='Explains technical questions in beginner-friendly English.',
        tags=["technology", "education"],
        examples=['What is ETL?'],
    )],
)


def main():
    """Example: A client calls an A2A server containing an LLM agent. 06_02_A2A_client.py: Retrieves the agent card and sends an A2A request. 06_02_A2A_server.py: Exposes an LLM agent to clients. This program: 06_02_A2A_server.py. Run: uv run python 06_02_A2A_server.py."""
    handler = DefaultRequestHandler(agent_executor=ExpertAgentExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler).build()
    print(f'Agent card: {SERVER_URL}.well-known/agent-card.json')
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
