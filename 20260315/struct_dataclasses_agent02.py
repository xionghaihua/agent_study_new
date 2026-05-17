from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from dotenv import load_dotenv
from dataclasses import dataclass
import os
load_dotenv()

#缺点：没有运行时验证

@dataclass
class BookInfo:
    title: str
    author: str
    isbn: str
    year: int
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
请严格按照以下要求，从记录中提取信息，**只返回标准JSON，不要任何其他文字**：
书信息：<百年孤独>加西亚.马尔克斯，ISBN9787544291170，1967年出版
必须返回的JSON格式：
{
    "title": "书名",
    "author": "作者",
    "isbn": "书编号",
    "year": "出版时间"
}
"""

book_agent=create_agent(
    model=model,
    tools=[placeholder_tool],
    response_format=ProviderStrategy(BookInfo)
)
result = book_agent.invoke({"messages":[{"role":"user","content":prompt}]})
book_data=result['structured_response']
print(book_data)
print(f"书名:{book_data.title}")