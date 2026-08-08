"""
Command是langgraph中用于控制图执行流程，更新图状态，并支持人机交互，工具调用的标准化对象
核心作用：
更新图的运行状态（更新state）
控制图的执行流向
衔接中断恢复，工具调用，人机交互等场景

command参数拆解：
update： 应用状态更新
goto： 导航到特定节点
graph： 从子图导航定位到父图
resume： 在中断后提供一个值以继续执行

在什么情况下使用
从节点返回：使用update，goto，graph 从状态更新与控制流结合
interrupt（人机交互）： 使用resume恢复
从工具返回





"""


from typing import Literal,TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command,Send

class MyState(TypedDict):
    type:str
    text:str
    result:str

#创建节点
def node_a(state: MyState):
    return {"result":"目前走了节点A"}

def node_b(state: MyState):
    return {"result":"目前走了节点B"}
def node_default(state: MyState):
    return {"result":"目前走了默认节点"}

#添加条件节点,Literal给当前返回值设置一个限制，只能返回literal列举的值
def judge_node(state: MyState)->Command[Literal["a","b","default"]]:
    if state["type"] == "a":
        return Command(update={"text":"走了A节点"},goto="a")  #Command可以更新状态
    elif state["type"] == "b":
        return Command(update={"text":"走了B节点"},goto="b")
    else:
        return Command(update={"text":"走了默认节点"},goto="default")

graph = StateGraph(state_schema=MyState)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_node("default", node_default)
graph.add_node("judge_node", judge_node)

graph.add_edge(START,'judge_node')
graph.add_edge("a",END)
graph.add_edge("b",END)
graph.add_edge("default",END)

app=graph.compile()
result = app.invoke({"type":"a"})
print(result)
