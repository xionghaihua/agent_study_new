from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.5-122b-a10b"
)

prompt = PromptTemplate(
    template="你是一位专业的程序员,\n对于信息: {text} 进行剪短描述"
)

input  = prompt.format(text="python")
res = model.invoke(input)
print(res.content)
