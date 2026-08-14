"""
不同状态模式
父图和子图有不同的模式，状态模式中没有共享的状态键，在这种情况下，必须在父图的节点内部调用子图
这是父图和子图有不同状态模式且需要在调用子图前后转换状态时很有用
"""

from langgraph.graph import StateGraph,MessagesState,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import  add_messages
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="openai:qwen3.7-plus",
    temperature=0
)
#创建子图
class SubgraphMessageState(TypedDict):
    subgraph_message: Annotated[list[AnyMessage],add_messages]

def subplot(state: SubgraphMessageState):
    answer = state["subgraph_message"][-1].content
    summary_prompt = f"请用一句话总结下面这段内容:\n\n答：{answer}"
    response = llm.invoke(summary_prompt)
    print("子图中问题和输出:",state["subgraph_message"] + [ response])
    return {"subgraph_message": [ response]}
summary_subgraph = (
    StateGraph(SubgraphMessageState)
    .add_node("subplot",subplot)
    .add_edge(START,"subplot")
    .compile()
)

#创建父图
def llm_answer_node(state:MessagesState)->MessagesState:
    answer = llm.invoke(state["messages"])
    print("父图中问题和输出:",state["messages"] + [ answer])
    #转换状态格式
    summary_result = summary_subgraph.invoke({"subgraph_message": state["messages"] + [ answer]})
    return {"messages": state["messages"] + [ answer] + [summary_result["subgraph_message"][2]]}

parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer_node",llm_answer_node)
    .add_edge(START,"llm_answer_node")
    .compile()
)
input_state = {
    "messages": [{"role": "user", "content": "langgraph是什么？"}]
}
result = parent_graph.invoke(input_state)
print("最终结果：",result)