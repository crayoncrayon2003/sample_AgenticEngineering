import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Part, TaskState, TextPart
from a2a.utils import new_task
from dotenv import load_dotenv
from langchain.agents import create_agent

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
HOST = "127.0.0.1"
PORT = 9000
SERVER_URL = f"http://{HOST}:{PORT}/"


def make_llm(effort: str):
    """LLM_PROVIDER に応じて、OpenAI または Claude のチャットモデルを作る。

    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
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
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5",
            reasoning_effort="low" if effort == "minimal" else effort,
            max_tokens=16000,
            timeout=60,
            max_retries=1,
        )
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


llm = make_llm("minimal")

#####################################
# 公開するエージェント：技術解説担当
#####################################
agent_expert = create_agent(
    # model　　　　　: エージェントに設定するLLM
    model=llm,
    # tools         : エージェントに設定するツール（今回は無し）
    tools=[],
    # system_prompt : エージェントに設定する役割
    system_prompt="あなたは技術解説の担当です。依頼された内容を、初学者向けの日本語で3文以内で説明してください。",
)


#####################################
# A2Aの依頼を、エージェントに渡す処理
#####################################
class ExpertAgentExecutor(AgentExecutor):
    # A2Aで依頼が届いたときに呼ばれる
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 依頼をタスクとして登録し、「作業中」にする
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.working)

        # エージェントを実行する
        result = await agent_expert.ainvoke({"messages": [{"role": "user", "content": context.get_user_input()}]})
        answer = result["messages"][-1].text

        # 回答を成果物（artifact）として登録し、タスクを「完了」にする
        await updater.add_artifact(parts=[Part(root=TextPart(text=answer))], name="answer")
        await updater.complete()

    # 依頼の取り消しには対応しない
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("取り消しには対応していません。")


#####################################
# エージェントカード：エージェントの名前・能力・接続先を、外部に公開する名刺
#####################################
agent_card = AgentCard(
    name="Technical Explanation Agent",
    description="技術的な質問を、初学者向けの日本語で説明するエージェントです。",
    url=SERVER_URL,
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[AgentSkill(
        id="explain_technical_topic",
        name="技術解説",
        description="技術的な質問を、初学者向けの日本語で説明します。",
        tags=["technology", "education"],
        examples=["ETLとは何ですか？"],
    )],
)


def main():
    """
    題材：クライアントが、A2Aサーバ（LLMを含むエージェント）を呼び出す。
         06_02_A2A_client.py : クライアント。エージェントカードを取得し、A2Aでエージェントに依頼する。
         06_02_A2A_server.py : A2Aサーバ。LLMを含むエージェントを公開し、クライアントから呼び出される。
         本プログラム         ： 06_02_A2A_server.py
         起動方法             : uv run python 06_02_A2A_server.py
    """
    handler = DefaultRequestHandler(agent_executor=ExpertAgentExecutor(), task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler).build()
    print(f"エージェントカード：{SERVER_URL}.well-known/agent-card.json")
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
