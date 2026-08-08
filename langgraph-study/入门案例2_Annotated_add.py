from typing import TypedDict,Annotated
from langgraph.graph import StateGraph
from operator import add

class MyState(TypedDict):
    names: Annotated[list[str],add]

def A_node(state:MyState):
    return {"names":["张三"]}
def B_node(state:MyState):
    return {"names":["李四"]}
state_graph = StateGraph(state_schema=MyState)
state_graph.add_node('A',A_node)
state_graph.add_node('B',B_node)

state_graph.set_entry_point("A")
state_graph.set_entry_point("B")

graph = state_graph.compile()
result = graph.invoke({})
print(result)  #{'names': ['张三', '李四']}


