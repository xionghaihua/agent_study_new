

from langgraph.graph import StateGraph,MessagesState,START,END
from langchain.chat_models import init_chat_model
import os
import uuid
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

#初始化检查点
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer",llm_answer_node)
    .add_node("summarize_subgraph",summary_subgraph)  #嵌套子图
    .add_edge(START,"llm_answer")
    .add_edge("llm_answer","summarize_subgraph")
    .add_edge("summarize_subgraph",END)
    .compile(checkpointer=checkpointer)
)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id":thread_id}}
input_state = {
    "messages": [{"role":"user","content":"langgraph是什么？,请用100字介绍"}],
}
result = parent_graph.invoke(input_state,config=config)
print(result)

#获取状态 get_state方法，获取当前任务所有保存的内容
current_state = parent_graph.get_state(config=config)
print(current_state)

#获取状态历史记录 get_state_history,
print("\n=============获取状态历史记录 get_state_history============")
history = list(parent_graph.get_state_history(config=config))
for idx,snapshot in enumerate(history):
    print(f"Step {idx}:")
    print(f"Checkpoint ID:{snapshot.config['configurable']['checkpoint_id']}")
    print(f"Node:{snapshot.metadata.get('source')}")
    print(f"messages: { [ m.content for m in snapshot.values['messages']]}")
    print("")
"""
Step 0:  #总结后
Checkpoint ID:1f193b64-e1b7-6282-8002-a1c5431b288b
Node:loop
messages: ['langgraph是什么？,请用100字介绍', 'LangGraph是LangChain推出的框架，用于构建有状态的复杂大模型应用。它通过图结构编排AI代理与多代理工作流，支持循环、条件分支和状态持久化。相比线性链，它提供更精细的流程控制，是开发高级AI Agent的核心工具。', 'LangGraph是LangChain推出的核心框架，通过图结构编排工作流并提供精细的流程控制，专门用于构建有状态的复杂大模型应用与高级AI Agent。']

Step 1:
Checkpoint ID:1f193b64-9467-64a0-8001-fcca957c524a
Node:loop
messages: ['langgraph是什么？,请用100字介绍', 'LangGraph是LangChain推出的框架，用于构建有状态的复杂大模型应用。它通过图结构编排AI代理与多代理工作流，支持循环、条件分支和状态持久化。相比线性链，它提供更精细的流程控制，是开发高级AI Agent的核心工具。']

Step 2:
Checkpoint ID:1f193b63-bed6-66ec-8000-3c7442f9126c
Node:loop
messages: ['langgraph是什么？,请用100字介绍']

Step 3:
Checkpoint ID:1f193b63-bed4-6022-bfff-4cec3bfefb14
Node:input
messages: []
"""
print("\n=======重放机制===============")
#任务执行完成之后，将中间的重要步骤给用户查看，用户就可以决定从某些步骤中重新开始执行
#注意，必须传递这些内容：thread_id，checkpoint_id
#通过thread_id 定位到某次会话
#通过checkpoint_id定位到某次对应的节点

#获取step2当前的checkpoint_id
step2_level_checkpoint = None
if history:
    #获取checkpoint_id
    step2_level_checkpoint = list(history)[2].config['configurable']['checkpoint_id']
#重新创建config内容，其中指定checkpoint_id，就可以从指定的checkpoint_id开始执行
config = {"configurable": {"thread_id":thread_id,"checkpoint_id":step2_level_checkpoint}}
new_result = parent_graph.invoke(None,config=config)
print(f"重播后的内容:{new_result}")

