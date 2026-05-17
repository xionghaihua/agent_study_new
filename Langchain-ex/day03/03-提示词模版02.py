#聊天提示词模版

from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

chat_prompt = ChatPromptTemplate.from_messages(
    [("system","你是一个数学家，你可以计算任何公式"),
    ("human","{text}")]
)

#print(chat_prompt)
#messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='你是一个数学家，你可以计算任何公式'), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['text'], input_types={}, partial_variables={}, template='{text}'), additional_kwargs={})]

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.5-122b-a10b"
)
message = chat_prompt.format_messages(text="我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁？")
res = model.invoke(message)
print(res.content)
