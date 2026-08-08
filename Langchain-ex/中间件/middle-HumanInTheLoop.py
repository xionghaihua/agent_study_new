#人机交互
"""
在工具调用执行前，暂停代理程序的执行，以便人工审批、编辑或拒绝这些调用。人机协作机制在以下情况下非常有用：
需要人工批准的高风险操作（例如数据库写入、金融交易）。
需要人工监督的合规工作流程。
长时间的对话，其中人类的反馈会指导智能体。

核心机制
安全守护：防止AI助手执行可能有害的操作
人工监督： 保持人类对关键决策的最终控制权
灵活决策： 支持批准，编辑，拒绝三种
审计追踪： 完整记录所有人工介入决策过程


智能监控机制：在调用工具之前，中间件通过智能测量识别需要人工介入的操作
自动批准： 安全的制度操作，如查询数据，读取文件
需要审核：写操作，删除操作，系统配置变更等


"""
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel,Field
from typing import Literal
from langchain.tools import tool
from dotenv import load_dotenv
import os
load_dotenv()

class WeatherInput(BaseModel):
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result

model = init_chat_model(
    base_url=os.getenv('ARK_BASE_URL'),
    api_key=os.getenv('ARK_API_KEY'),
    model_provider="openai",
    model="Doubao-Seed-2.0-lite"
)

#创建人工介入的中间件
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "write_file": True, #所有决策类型允许
        "execute_sql": {"allowed_decisions":["approve","reject"]},
        "read_data": False, #自动批准
        "get_weather": {"allowed_decisions":["approve","reject"]},
    },
    description_prefix="工具执行待审核"
)

agent = create_agent(
    model=model,
    tools = [get_weather],
    middleware=[hitl_middleware],
    checkpointer = InMemorySaver()
)

from langgraph.types import Command
config = {"configurable":{"thread_id":"con_123"}}
result = agent.invoke(
    {"messages":[{"role":"user","content":"当前天气情况"}]},
    config = config
)
#检测是否有中断
