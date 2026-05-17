from langchain.agents.middleware import after_model,AgentState
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langgraph.runtime import Runtime
from langchain.chat_models import init_chat_model
from typing import Any
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
load_dotenv()

@after_model
def format_response(state:AgentState,runtime:Runtime)->Any:
    #分析最后一条AI消息
    if state["messages"] and isinstance(state["messages"][-1],AIMessage):
        last_message = state["messages"][-1]
        if "```" in last_message.content:
            enhanced_content = last_message.content + "\n\n提示：以上代码仅供参考，请根据实际需求进行调整。"
            # 创建新的消息列表，只替换最后一条消息，保留其他所有消息
            new_messages = state["messages"][:-1] + [
                AIMessage(content=enhanced_content, name=last_message.name)
            ]
            return {"messages": new_messages}
        return None
#配置模型
model = init_chat_model(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
    temperature=0.1,
    timeout=60,
    max_tokens=2000,
    model_provider="openai"
)
smart_agent=create_agent(
    model=model,
    middleware=[format_response]
)
result = smart_agent.invoke({
        "messages": [HumanMessage("请写一个hello world的python代码")]
    },
    user_id="123456"
)

# 只输出最后一条消息（AI的响应）
if result["messages"]:
    print(result["messages"][-1].content)
    print("######################################")