#少量样本数据的提示词模版

from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

examples = [
    {"input": "2+2", "output": "4", "description":" 加法运算"},
    {"input": "5-2", "output": "3", "description": "减法运算"},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output", "description"],
    template="问题：{input}\n类型：{description}\n答案：{output}"
)
# 3. 构建少样本提示模板
prompt = FewShotPromptTemplate(
    examples=examples,                # 传入示例
    example_prompt=example_prompt,   # 示例的格式
    suffix="问题：{input}\n类型：{description}\n答案：{output}",  # 最后要提问的格式
    input_variables=["input"],       # 最终输入变量
)

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.5-122b-a10b"
)
result = model.invoke(prompt.format(input="2*5",output="10",description="乘法运算"))
print(result.content)