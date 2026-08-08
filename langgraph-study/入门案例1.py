from typing import TypedDict
from langgraph.graph import StateGraph
#定义全局State，TypedDict推荐使用
class MyState(TypedDict):
    question: str
    answer: str

#工作节点
def llm_result(state: MyState):
    print(state["question"])
    return {"answer":"这是答案"}

#初始化状态图
state_graph = StateGraph(state_schema=MyState)
#添加节点
#参数1:节点名称
#节点2:节点函数
state_graph.add_node("llm_result", llm_result)
#定义入口
state_graph.set_entry_point("llm_result")

#构造图
graph = state_graph.compile()

#执行图
result = graph.invoke({"question":"你是谁?"})
print(result)