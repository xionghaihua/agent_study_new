#大语言模型
from langchain_community.llms import Tongyi
from dotenv import load_dotenv
import os

load_dotenv()

"""
llm = Tongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
)

text = "真的好想(帮我补齐这个文本)"
res = llm.invoke(text)
print(res)
"""
#聊天模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
)
text = "真的好想(帮我补齐这个文本)"
res = llm.invoke(text)
print(res.content)



