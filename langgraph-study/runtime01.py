#运行时配置Runtime
"""
创建图的时候，可以标记图的某些部分是可配置的，这样做通常是为了方便在模型或系统提示之间切换
在运行图时提供额外的配置参数而不是状态参数，并通过类型约束这些参数
"""
from dataclasses import dataclass

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing import TypedDict
import dataclasses

class MyState(TypedDict):
    question:str
    answer:str

#定义配置结构
@dataclasses.dataclass(frozen=True)
class MyContext(TypedDict):
    language:str

#节点函数可以访问runtime参数，runtime可以访问上下文和内存存储

def step(state: MyState,runtime:Runtime[MyContext]):
    if runtime.context["language"] == "zh":
        answer = "您好"
    else:
        answer = "hello"
    return {"answer":answer}

graph = StateGraph(state_schema=MyState,context_schema=MyContext)
graph.add_node("step1",step)
graph.set_entry_point("step1")

app = graph.compile()
#context
result = app.invoke({"question":"Hi"},context={"language":"zh"})
print(result['answer'])