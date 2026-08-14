"""
持久性

检查点checkpointing是langgraph持久性的核心机制，它允许你在图执行过程中保存状态

核心概念：
检查点：图状态的快照
线程： 用于访问检查点的唯一标识，一次会话
检查点保存checkponter:负责保存和恢复状态的组件

线程threads： 线程是检查点保存器保存的每个检查点分配的唯一ID
config = {"configurable": {"thread_id":"UUID"}}
result = graph.invoke(input_data,config=config)

特点：
每个线程代表一个独立的对话或执行上下文
线程允许在图执行后访问图的状态
执行多个并发线程

检查点：在每个超级步骤中保存图状态的快照。
config: 与检查点相关的配置
metadata： 与检查点相关的元数据
values: 当前state的值，也就是图执行到目前为止，所有变量的值，如messages，steps，results
next: 接下来要执行的节点名称的元组
tasks：具体要执行的任务的详细信息






"""
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
import uuid
load_dotenv()


#初始化模型
llm = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="openai:qwen3.7-plus",
    temperature=0
)

#自定义图状态
class MyState(MessagesState):
    result:str

#定义相关的node
async def process_message(state:MyState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": response}

async def optimize_message(state:MyState):
    messages = state["messages"] + [{"role":"system","content":"请用幽默的形式回复用户"}]
    response = await llm.ainvoke(messages)
    return {"messages": response}

#使用StateGraph构建图
builder = StateGraph(state_schema=MyState)
builder.add_node("process_message",process_message)
builder.add_node("optimize_message",optimize_message)
builder.add_edge(START,"process_message")
builder.add_edge("process_message","optimize_message")
builder.add_edge("optimize_message",END)
#初始化检查点
checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

async def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id":thread_id}}
    print(f"当前会话thread_id:{thread_id}")

    input_message = {"messages":[HumanMessage(content="您好，我叫alex")]}
    result = await app.ainvoke(input_message,config=config)
    print(result["messages"][-1].content)
    #获取最新检查点快照
    snapshot = await app.aget_state(config)
    print(f"当前state的values:{snapshot.values}")
    print(f"下一个待执行节点:{snapshot.next}")


    input_message1 = {"messages":[HumanMessage(content="你还知道我是谁吗？")]}
    result1 = await app.ainvoke(input_message1, config=config)
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())




