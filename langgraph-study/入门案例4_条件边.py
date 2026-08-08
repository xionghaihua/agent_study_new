from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import END
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
load_dotenv()
langfuse_handler=CallbackHandler()
class MyState(TypedDict):
    type: str
    result: str

def judge_node(state:MyState):
    return state

def route_condition(state:MyState):
    if state["type"] == "a":
        return "a"
    elif state["type"] == "b":
        return "b"
    else:
        return "default"

def node_a(state):
    return {"result":"走了A分支"}
def node_b(state):
    return {"result":"走了B分支"}
def node_default(state):
    return {"result":"走了默认分支"}

graph = StateGraph(state_schema=MyState)
graph.add_node("judge_node",judge_node)
graph.add_node("a",node_a)
graph.add_node("b",node_b)
graph.add_node("default",node_default)
#设置开始
graph.set_entry_point("judge_node")
#条件,条件边
graph.add_conditional_edges(
    "judge_node",
    route_condition,
    {"a":"a","b":"b","default":"default"},
)
#结束边
graph.add_edge("a",END)
graph.add_edge("b",END)
graph.add_edge("default",END)

app = graph.compile()
result = app.invoke({"type":"b","result":""},config={"callbacks":[langfuse_handler]})
print(result)