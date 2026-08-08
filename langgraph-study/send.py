from typing import List, Annotated, TypedDict
import operator
# 修复Send导入位置
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    numbers: List[int]
    results: Annotated[list[int], operator.add]
    final_sum: int  #最终求和


#map阶段 分发数字【节点函数：只返回状态，不再返回Send数组！】
def split_numbers(state: State):
    """把数字保存到state，路由函数负责派发"""
    numbers = state["numbers"]
    print(f"分发数字:{numbers}")
    return {}  # 不需要修改状态，返回空dict


# 👉【新增条件路由函数】专门生成Send并行任务
def route_to_workers(state: State):
    numbers = state["numbers"]
    return [Send("worker", {"number": num}) for num in numbers]


# worker阶段，计算平方
def calculate_square(state: State):
    number = state["number"]
    square = number * number
    print(f"worker:{number}的平方 = {square}")
    return {"results": [square]}


# Reduce阶段
def sum_results(state: State):
    results = state.get("results", [])
    total = sum(results)
    print(f"求和：{results} = {total}")
    return {"final_sum": total}


#构建图
def create_simple_graph():
    graph = StateGraph(State)
    #添加节点
    graph.add_node("splitter", split_numbers)
    graph.add_node("worker", calculate_square)
    graph.add_node("sum", sum_results)

    graph.add_edge(START, "splitter")

    # ✅ 条件边：路由函数route_to_workers负责生成Send
    graph.add_conditional_edges(
        source="splitter",
        path=route_to_workers,
        path_map=["worker"]
    )

    graph.add_edge("worker", "sum")
    graph.add_edge("sum", END)

    return graph.compile()


def main():
    app = create_simple_graph()
    initial_state = {
        "numbers": [1, 2, 3, 4, 5],
        "results": [],
        "final_sum": 0
    }
    result = app.invoke(initial_state)
    print("\n最终结果：", result)


if __name__ == "__main__":
    main()