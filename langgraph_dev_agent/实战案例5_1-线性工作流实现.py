#定义状态
"""
定义状态State
状态是工作流的数据总线，需明确村粗，输入数据--中间结果--输出数据，langgraph支持两种状态定义方式
简单场景：TypedDict定义强类型状态
复杂场景：使用pedantic.BaseModel定义带校验的状态
"""
from typing import TypedDict, Optional

#定义状态，存储原始文档，预处理后的文档，LLM摘要结果
class SummaryState(TypedDict):
    raw_document:str
    processed_document:str
    summary:Optional[str]

#实现节点逻辑
"""
节点是工作流的执行单元，每个节点接收State作为输入，修改后，并返回新的State
"""
#节点1:文档预处理
def process_document(state:SummaryState)->SummaryState:
    """预处理原始文档"""
    raw_doc = state["raw_document"]
    processed = raw_doc.strip().replace("\n\n","\n").replace("  "," ")
    #返回更新状态
    return {"processed_document":processed}
#节点2: DeepSeek LLM摘要生成
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.7-plus",
    temperature=0
)
#定义摘要提示词模版
summary_prompt = ChatPromptTemplate.from_messages([
    ("system","你是专业的文本摘要助手，需基于以下文本生成简洁、准确的摘要，不添加额外信息，长度控制在300字以内。"),
    ("human",f"文本内容：{process_document}")
])
#构建摘要链
summary_chain = summary_prompt | llm | StrOutputParser()

def generate_summary(state:SummaryState)->SummaryState:
    """调用LLM生成文本摘要"""
    processed_document = state["processed_document"]
    summary = summary_chain.invoke({"processed_text":processed_document})
    #返回更新后的状态
    return {"summary": summary}

#节点3: 摘要后处理（格式化）
def format_summary(state:SummaryState)->str:
    """后处理摘要结果，格式化"""
    summary = state["summary"]
    #格式化
    formated_summary = f"###文档摘要\n\n{summary.replace('. ','.\n')}"
    return {"summary": formated_summary}

#构建线性工作流图（StateGraph）
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
builder = StateGraph(SummaryState)
#添加节点
builder.add_node("process_doc",process_document) #节点1:预处理
builder.add_node("generate_summary",generate_summary)#节点2: LLM摘要
builder.add_node("summary",format_summary)  #节点3:后处理，格式化
#定义线性边
builder.add_edge("process_doc","generate_summary")
builder.add_edge("generate_summary","summary")
builder.add_edge("summary",END)
#设置入口节点
builder.set_entry_point("process_doc")
#编译生成工作流
summary_graph = builder.compile(checkpointer=checkpointer)

#4、执行工作流
raw_document = """
LangGraph 是 LangChain 生态系统中的一个框架，专门用于构建状态ful、可循环的工作流。
它基于状态机的思想，允许开发者定义节点和边，并通过状态对象管理整个工作流的数据流转。
与传统的线性脚本相比，LangGraph 提供了更好的可扩展性和可观测性，特别适合 LLM 应用中的复杂流程编排，
例如多轮对话、文档分析、工具调用链等场景。LangGraph 的核心组件包括 StateGraph、State、Node 和 Edge，
这些组件共同构成了灵活且强大的工作流系统。
"""
config={"configurable":{"thread_id":"summary_01"}}
result = summary_graph.invoke({"raw_document":raw_document},config=config)

print("最终结果:")
print(result["summary"])

