"""
LLM API超时
数据库连接抖动
网络请求失败

retry_policy
"""


import sqlite3
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState,StateGraph,START,END
from langgraph.types import RetryPolicy
from langchain_community.utilities import SQLDatabase
from langchain.messages import AIMessage,HumanMessage
from langgraph.runtime import Runtime
from dotenv import load_dotenv
import os
load_dotenv()

db = SQLDatabase.from_uri("sqlite:///:memory:")
model = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="openai:qwen3.7-plus",
    temperature=0
)

#自定义异常
def query_database(state:MessagesState,runtime:Runtime):
    print(f"正在尝试第{runtime.execution_info.node_attempt}次查询")
    #获取当前的重试次数runtime.execution_info.node_attempt
    if runtime.execution_info.node_attempt < 3:
        print("模拟数据库连接失败")
        raise sqlite3.OperationalError("Database connection error")
    query_result = db.run("SELECT 1;")
    return {"messages":[AIMessage(content=str(query_result))]}


def call_model(state:MessagesState,runtime:Runtime):
    response = model.invoke(state["messages"])
    return {"messages":[response]}

builder = StateGraph(MessagesState)
builder.add_node(
    "query_database",
    query_database,
    retry_policy=RetryPolicy(retry_on=[sqlite3.OperationalError,sqlite3.IntegrityError]),
)

builder.add_node("model",call_model,retry_policy=RetryPolicy(max_attempts=5))
builder.add_edge(START,"model")
builder.add_edge("model","query_database")
builder.add_edge("query_database",END)
graph = builder.compile()

response = graph.invoke({"messages":[HumanMessage(content="你好啊？")]})
print(response["messages"])

