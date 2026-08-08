from typing import TypedDict,Annotated
import operator
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command

class State(TypedDict):
    num: int
    msg: str
def large_handler(state: State):
    return {}
def small_handler(state: State):
    return {}

def check_num(state: State) ->Command:
    n=state["num"]
    if n>5:
        return Command(
            update={"msg":"数字大于5"},
            goto="large_handler"
        )
    else:
        return Command(
            update={"msg":"数字小于等于5"},
            goto="small_handler"
        )
builder = StateGraph(State)
builder.add_node("check_num", check_num)
builder.add_node("large_handler", large_handler)
builder.add_node("small_handler", small_handler)
builder.add_edge(START, "check_num")
# ✅ 重点：不需要 add_conditional_edges！Command内部完成路由
builder.add_edge("large_handler", END)
builder.add_edge("small_handler", END)

graph = builder.compile()
res = graph.invoke({"num": 8})
print(res)