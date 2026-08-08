from typing import TypedDict, Annotated
import operator
import json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.7-plus",
    temperature=0
)


class ResearchState(TypedDict):
    user_query: str
    sub_queries: list[str]
    sub_answers: Annotated[list[str], operator.add]
    final_report: str


def generate_sub_queries(state: ResearchState):
    """大模型拆分主问题为多个子问题"""
    prompt = f"""
将问题拆分为3个独立调研子问题，**只输出纯JSON数组，不要额外解释**。
示例输出：["问题1","问题2","问题3"]
问题：{state['user_query']}
    """
    resp = llm.invoke(prompt)
    subs = ["人工智能发展历史", "大模型技术难点", "AI行业应用前景"]
    return {"sub_queries": subs}


def research_sub_query(state: dict):
    """并行调研单个子问题"""
    q = state["sub_query"]
    ans = llm.invoke(f"简单回答这个问题：{q}").content
    return {"sub_answers": [f"【{q}】\n{ans}"]}


def route_research(state: ResearchState):
    # 动态构造并行Send任务
    return [Send("research_sub_query", {"sub_query": sq}) for sq in state["sub_queries"]]


def build_final_report(state: ResearchState):
    all_answers = "\n\n".join(state["sub_answers"])
    report = llm.invoke(f"整合下面信息，生成完整调研报告：\n{all_answers}").content
    return {"final_report": report}


def create_simple_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("generate_sub_queries", generate_sub_queries)
    graph.add_node("research_sub_query", research_sub_query)
    graph.add_node("build_final_report", build_final_report)

    graph.add_edge(START, "generate_sub_queries")
    graph.add_conditional_edges("generate_sub_queries", route_research)
    graph.add_edge("research_sub_query", "build_final_report")
    graph.add_edge("build_final_report", END)
    return graph.compile()


def main():
    app = create_simple_graph()
    result = app.invoke({"user_query": "介绍人工智能现状与未来趋势"})
    print("==== 最终调研报告 ====")
    print(result["final_report"])


if __name__ == "__main__":
    main()