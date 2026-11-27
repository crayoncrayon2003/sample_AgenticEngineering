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

# --- 設定 ---
ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, ".env"))
PROVIDER = os.getenv("LLM_PROVIDER") or "openai"  # openai または anthropic
HOST = "127.0.0.1"
PORT = 9000
SERVER_URL = f"http://{HOST}:{PORT}/"


def make_llm(effort: str) -> dict:
    """LLM_PROVIDER に応じて、LiteLLM に渡すモデルの設定を作る。

    LiteLLM は「提供元/モデル名」の文字列で、呼び出すLLMを切り替える。
    effort：推論の深さ。Claude には minimal が無いため、low として扱う。
    """
    if PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に OPENAI_API_KEY を設定してください。")
        return {
            "model": "openai/" + (os.getenv("OPENAI_MODEL") or "gpt-5-nano"),
            "reasoning_effort": effort,
            "max_tokens": None,
        }
    if PROVIDER == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(f"{os.path.join(ROOT, '.env')} に ANTHROPIC_API_KEY を設定してください。")
        return {
            "model": "anthropic/" + (os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"),
            "reasoning_effort": "low" if effort == "minimal" else effort,
            "max_tokens": 16000,
        }
    raise RuntimeError("LLM_PROVIDER は openai か anthropic を指定してください。")


def call_llm(llm: dict, messages: list, response_format=None, tools=None):
    """LiteLLM でLLMを呼び、返ってきたメッセージを返す（LangChain の llm.invoke に当たる）。"""
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
    """プロンプトの {名前} に値を入れる（LangChain の ChatPromptTemplate に当たる）。"""
    return [{"role": m["role"], "content": m["content"].format_map(values)} for m in prompt]


llm = make_llm("minimal")

#####################################
# 公開するエージェント：技術解説担当
#####################################
# system_prompt : エージェントに設定する役割
system_prompt = "あなたは技術解説の担当です。依頼された内容を、初学者向けの日本語で3文以内で説明してください。"


# 技術解説担当に依頼し、回答を返す（A2Aのサーバーは非同期で動くため、litellm.acompletion を使う）
async def agent_expert(message: str) -> str:
    response = await litellm.acompletion(
        model=llm["model"], reasoning_effort=llm["reasoning_effort"], max_tokens=llm["max_tokens"],
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": message}],
        timeout=60, num_retries=1,
    )
    return response.choices[0].message.content


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
        answer = await agent_expert(context.get_user_input())

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
