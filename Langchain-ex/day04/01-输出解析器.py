from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser,XMLOutputParser
from langchain_classic.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()


llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)
xml_parser = XMLOutputParser()
json_parser = JsonOutputParser()
str_parser = StrOutputParser()
prompt = ChatPromptTemplate([
    ("system","你是一个专业的程序员"),
    ("user","{input}")
])

#0.2,0.3版本
"""
chain = LLMChain(
    prompt=prompt,
    llm=llm,
    #output_parser=xml_parser,
    output_parser=json_parser,
)

"""
#

chain = prompt | llm | json_parser

#pip install defusedxml
res = chain.invoke({"input":"langchain是什么？,使用json格式返回结果"})
print(res)
"""
{'description': 'LangChain是一个用于开发基于大语言模型（LLM）的应用的开源框架。它提供了模块化的组件（如提示模板、链、代理、记忆等）来简化LLM与外部数据源、工具和API的集成，支持构建对话系统、文档问答、数据增强生成等复杂工作流。'}
"""
