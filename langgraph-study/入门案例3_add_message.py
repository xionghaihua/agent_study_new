from typing import TypedDict,Annotated
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

class ChatState(MessagesState):
    messages: Annotated[list[AnyMessage],add_messages]

def user_input_node(state:ChatState)->dict:
    user_msg = {"role":"user","content":"什么是langgraph？"}
    return {"messages":[user_msg]}

def assistant_node(state:ChatState)->dict:
    reply = {"role": "assistant", "content": "Langgraph是一个有状态的图编排工具"}
    return {"messages":[reply]}

builder = StateGraph(state_schema=ChatState)
builder.add_node("user_input",user_input_node)
builder.add_node("assistant_node",assistant_node)
builder.set_entry_point("user_input")
builder.add_edge("user_input","assistant_node")

state = builder.compile()
result = state.invoke({})
print(result)


