from typing import Annotated,TypedDict,Literal
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command,Send
import operator
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
    sub_answers: Annotated[list[str],operator.add]
    final_report:str
def generate_sub_queries(state:ResearchState) -> Command[Literal["research_sub_query"]]:
    prompt = f"""
    将问题拆分为3个独立调研子问题，**只输出纯JSON数组，不要额外解释**。
    示例输出：["问题1","问题2","问题3"]
    问题：{state['user_query']}
        """
    resp = llm.invoke(prompt)

    subs = ["人工智能发展历史", "大模型技术难点", "AI行业应用前景"]
    # 使用Command：更新state + 批量Send并行任务
    return Command(
        update={"sub_queries":subs},
        goto=[Send("research_sub_query",{"sub_query":sq}) for sq in subs],
    )

def research_sub_query(state: dict):
    """并行调研单个子问题"""
    q = state["sub_query"]
    ans = llm.invoke(f"简单回答这个问题：{q}").content
    return {"sub_answers": [f"【{q}】\n{ans}"]}

def build_final_report(state: ResearchState):
    all_answers = "\n\n".join(state["sub_answers"])
    report = llm.invoke(f"整合下面信息，生成完整调研报告：\n{all_answers}").content
    return {"final_report": report}

builder = StateGraph(ResearchState)
builder.add_node("generate_sub_queries", generate_sub_queries)
builder.add_node("research_sub_query", research_sub_query)
builder.add_node("build_final_report", build_final_report)

builder.add_edge(START, "generate_sub_queries")
builder.add_edge("research_sub_query", "build_final_report")
builder.add_edge("build_final_report", END)

graph = builder.compile()
result = graph.invoke({"user_query":"介绍人工智能现状与未来趋势"})
print(result["final_report"])
