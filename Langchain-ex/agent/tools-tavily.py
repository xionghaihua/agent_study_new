#pip install langchain-tavily
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

model=init_chat_model(
    base_url=os.getenv('ARK_BASE_URL'),
    api_key=os.getenv('ARK_API_KEY'),
    model_provider="openai",
    model="Doubao-Seed-2.0-lite",
    temperature=0
)
tools = [TavilySearch(max_results=1)]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个超级智能助手，能帮助用户解决问题"
)
query = "请问现任的美国总统是谁？他的年龄是多少？请用中文回答"
try:
    result = agent.invoke({"messages":[{"role":"user","content":query}]})
    print(result['messages'][-1].content)
except Exception as e:
    print(f"发生错误：{e}")

