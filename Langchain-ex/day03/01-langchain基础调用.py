from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()


llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.5-122b-a10b",
    temperature=0.2
)

res = llm.invoke("什么事大模型")
print(res.content)
print(res.response_metadata)