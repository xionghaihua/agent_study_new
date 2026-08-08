"""
当接近消息数量上限时，自动生成对话历史记录摘要，保留最近的消息，同时压缩较早的上下文。摘要功能适用于以下情况：
持续时间过长的对话，超出上下文窗口。
多轮对话，历史悠久。
需要保留完整对话上下文的应用场景。

摘要是一种面向文本的上下文压缩。它不会调整图像/音频/视频有效负载的大小、降采样或以其他方式压缩它们

核心价值：
上下文压缩：自动总结旧消息，释放token空间
信息保留：智能提取关键信息
对话连贯：保留多轮对话的连贯性和上下文理解
成本优化：减少不必要的token消耗



触发机制 （支持单一条件或多个条件组合）
token数量：当对话总token的数达到设定阈值时触发，使用trigger={"token",value}
消息数量： 当对话消息数量达到设定的阈值时触发，使用trigger={"messages",value}
上下文比例：当对话占用模型上下文窗口的比例达到设定的值时候触发，使用trigger={"fraction",value}

保留策略
消息保留：通过keep={"messages",value}参数指定保留的消息数量（默认20条）
token保留：通过keep={"token",value}参数指定保留的token数量
比例保留： 通过keep={"fraction",value}参数安比例保留上下文
智能提取：使用指定的模型提取关键实体、意图和上下文信息
总结生成：生成高质量的对话总结并添加到消息历史中
上下文维护

model：用于政策总结的模型
summary_prefix： 添加到总结消息的前缀文本，如summary_prefix=“对话摘要:"

token_counter: 用于计算对话消息的token数量的函数
自定义计数函数
def custom_token_counter(messages):
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4o")
    total_tokens=0
    for message in messages:
        content = message.get("content","") or ""
        total_token += len(encoding.encode(content))
    return total_tokens
token_counter = custom_token_counter

summary_prompt: 用于生成对话总结的提示模版
summary_prompt = None
summary_prompt="你是一位专业的对话总结助手，请总结以下对话内容，重点关注1.用户的主要问题和需求，请具体列出，2。已经提供的解决方案和结果
3.尚未解决的关键问题，4.重要的实体，参数和上下文信息，请以简洁明了的方式组织信息，确保保留所有关键细节。对话内容：{messages}"
"""
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
custom_profile = {
    "max_input_tokens": 100000,
}
#工具
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
#自定义总结提示模版
custom_summary_prompt="""
请总结以下对话内容,重点关注:
1.用户的主要问题和需求
2.已经提供的解决方案
3.尚未解决的关键问题
4.重点的上下文信息

对话内容：
{messages}
"""
model = init_chat_model(
    base_url=os.getenv('ARK_BASE_URL'),
    api_key=os.getenv('ARK_API_KEY'),
    model_provider="openai",
    model="Doubao-Seed-2.0-lite",
    profile=custom_profile,
)
custom_summarization = SummarizationMiddleware(
    model=model,
    #OR逻辑，只要满足其中一个就会触发总结
    trigger=[("tokens",4000),("messages",10)],
    keep=("messages",30),
    summary_prompt=custom_summary_prompt,
)
agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[custom_summarization],
)

result1 = agent.invoke(
    {"messages":[{"role":"user","content":"请你介绍一下langchain？"}]}
)
print(result1['messages'][-1].content)

#做总结的建议
"""
推荐使用轻量级模型进行总结，如gpt-4o-mini
合理设置触发条件，避免过度频繁的总结
保留足够的最近消息以确保上下文连贯
自定义总结提示以适应具体的业务场景
"""
