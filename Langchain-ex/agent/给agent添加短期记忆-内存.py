from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver  #基于内存的短期记忆
from dotenv import load_dotenv
import os

load_dotenv()
#配置llm
llm=init_chat_model(
    base_url=os.getenv('ARK_BASE_URL'),
    api_key=os.getenv('ARK_API_KEY'),
    model_provider="openai",
    model="Doubao-Seed-2.0-lite",
    temperature=0
)

#配置工具
tools=[TavilySearch(max_results=1)]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个超级智能助手，能帮助用户解决问题",
    checkpointer=InMemorySaver()
)

#执行任务
config={"configurable":{"thread_id":"agent_1"}}
query1 = "请问现任的美国总统是谁？他的年龄是多少？请用中文回答"
query2 = "请问我上一个问题问了什么？"

try:
    result1 = agent.invoke({"messages":[{"role":"user","content":query1}]},config=config)
    print(result1["messages"][-1].content)
    result2 = agent.invoke({"messages":[{"role":"user","content":query2}]},config=config)
    print(result2["messages"][-1].content)
except Exception as e:
    print(f"发生错误:{e}")

