from langchain.agents.structured_output import ProviderStrategy
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# 你的结构化输出模型
class MeetingAction(BaseModel):
    topic: str = Field(description="会议主题")
    participants: list[str] = Field(description="参会人员列表")  # 明确str类型
    action_item: list[str] = Field(description="行动项")
    deadline: str = Field(description="截止时间")

# 初始化模型
model = init_chat_model(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
    temperature=0.1,
    timeout=60,
    max_tokens=2000,
    model_provider="openai"
)

# 占位工具（必须传，解决空tools报错）
@tool
def placeholder_tool() -> str:
    """占位工具，无实际功能"""
    return "success"

prompt = """
请严格按照以下要求，从会议记录中提取信息，**只返回标准JSON，不要任何其他文字**：
会议记录：项目评审会议，张三和李四参加，需要完成代码审查，截止到下周三
必须返回的JSON格式：
{
    "topic": "会议主题",
    "participants": ["姓名1", "姓名2"],
    "action_item": ["任务1"],
    "deadline": "截止时间"
}
"""

# 创建 Agent
meeting_agent = create_agent(
    model=model,
    tools=[placeholder_tool],
    response_format=ProviderStrategy(MeetingAction)
)

# 调用
result = meeting_agent.invoke({
    "messages": [{"role": "user", "content": prompt}]
})

# 输出结果
meeting_data = result['structured_response']
print(meeting_data)
print("✅ 解析成功！")
print(f"会议主题: {meeting_data.topic}")
print(f"参会人员: {meeting_data.participants}")
print(f"行动项: {meeting_data.action_item}")
print(f"截止时间: {meeting_data.deadline}")