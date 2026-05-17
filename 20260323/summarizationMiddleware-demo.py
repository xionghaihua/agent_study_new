from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化聊天模型
model = init_chat_model(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
    temperature=0.1,
    timeout=60,
    max_tokens=2000,
    model_provider="openai"
)

# 创建内存检查点，用于保存对话历史
checkpointer = InMemorySaver()

# 创建SummarizationMiddleware实例
summarization = SummarizationMiddleware(
    model=model,
    trigger=[("tokens", 100), ("messages", 10)], # 超过100 tokens或10条消息时才触发总结（替代max_tokens_before_summary）
    keep=("messages", 2) # 保留最近2条消息（替代messages_to_keep）
)

# 创建带短期记忆功能的agent
agent = create_agent(
    model=model,
    checkpointer=checkpointer,      # 添加检查点实现会话持久化
    system_prompt="你是诚实的智能助手，你的回答总是诚实简洁。",
    middleware=[summarization] # 添加总结中间件
)
# 这里设置thread_id:1,告诉大模型自己叫小明。
result= agent.invoke({"messages": [{"role": "user", "content": "你好！我叫小明。酒精能杀灭病毒吗？"}]}, {"configurable": {"thread_id": "1"}},)

# 制造20轮对话
for i in range(20):
    result= agent.invoke({"messages": [{"role": "user", "content": "还记得我的名字吗？"}]}, {"configurable": {"thread_id": "1"}},)




# 看看多轮对话后汇总中间件的效果
result= agent.invoke({"messages": [{"role": "user", "content": "还记得我的名字吗？"}]}, {"configurable": {"thread_id": "1"}},)
print("\n".join([m.content for m in result["messages"]]))