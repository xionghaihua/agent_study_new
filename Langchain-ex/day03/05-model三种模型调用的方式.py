from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()

llm = init_chat_model(
    "deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

res = llm.invoke("什么是大模型")
#print(res.content)


#第二种方式
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-v4-flash",
    temperature=0
)
res1 = llm.invoke("什么是langchain")
#print(res1.content)


#第三种
#pip install langchain_deepseek

from langchain_deepseek import  ChatDeepSeek

llm = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-v4-flash",
    temperature=0
)
res2 = llm.invoke("什么是大语言模型")
print(res2.content)
