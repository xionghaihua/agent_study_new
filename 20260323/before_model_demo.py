from langchain.agents.middleware import before_agent,AgentState
from langchain.messages import SystemMessage,HumanMessage
from langgraph.runtime import Runtime
from langchain.chat_models import init_chat_model
from typing import Any
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
load_dotenv()
@before_agent(can_jump_to=["end"])
def filter_sensitive_content(state: AgentState,runtime:Runtime) ->dict[str,Any]|None:
    #检测是否包含敏感词内容
    sensitive_words = ["敏感词1","敏感词2"]
    for msg in state["messages"]:
        if isinstance(msg,HumanMessage):
            for word in sensitive_words:
                if word in msg.content.lower():
                    return {
                        "messages": [SystemMessage(content="检测到敏感内容，请重新输入。")],
                        "jump_to": "end"
                    }
    #截断过长的消息
    processed_messages = []
    for msg in state["messages"]:
        if len(msg.content) > 1000:
            msg.content = msg.content[:990]+ "...[内容已截断]"
        processed_messages.append(msg)
    return {"messages": processed_messages}
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
    middleware=[filter_sensitive_content]
)
result = smart_agent.invoke({
        "messages": [HumanMessage("敏感词1是什么意思？")]
    },
    user_id="123456"
)
# 转换所有消息为可序列化的字典
# 输出最后两条Messages
for msg in result["messages"]:
    print(msg.content)
    print("######################################")


