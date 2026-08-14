
"""
#共享状态键
父图和子图在其状态模式中有共享的状态键，在这种情况下，你可以将子图作为节点包含在父图中
"""

from langgraph.graph import StateGraph,MessagesState,START,END
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
def subplot(state:MessagesState)->MessagesState:
    #获取大模型回答的内容进行摘要总结
    answer = state["messages"][-1].content
    summary_prompt = f"请用一句话总结下面这段内容:\n\n答：{answer}"
    response = llm.invoke(summary_prompt)
    return {"messages": state["messages"] + [response]}

summary_subgraph = (
    StateGraph(MessagesState)
    .add_node("subplot",subplot)
    .add_edge(START,"subplot")
    .compile()
)

#创建父图
def llm_answer_node(state:MessagesState)->MessagesState:
    answer = llm.invoke(state["messages"])
    #print("父图输出",answer)
    return {"messages": state["messages"] + [answer]}
parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer",llm_answer_node)
    .add_node("summarize_subgraph",summary_subgraph)
    .add_edge(START,"llm_answer")
    .add_edge("llm_answer","summarize_subgraph")
    .add_edge("summarize_subgraph",END)
    .compile()
)

input_state = {
    "messages": [{"role":"user","content":"langgraph是什么？"}]
}
result = parent_graph.invoke(input_state)
print(result['messages'][-1].content)
