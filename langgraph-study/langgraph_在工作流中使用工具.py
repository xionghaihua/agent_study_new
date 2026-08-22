from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()

llm = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="openai:qwen3.7-plus",
    temperature=0
)

@tool
def tavily_search_tool(query:str)->str:
    """搜索工具"""
    tool_instance = TavilySearch()
    return tool_instance.run(query)
tool_node = ToolNode([tavily_search_tool])
model_with_tools = llm.bind_tools([tavily_search_tool])

def should_continue(state:MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END
def call_model(state:MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
builder = StateGraph(MessagesState)
builder.add_node("call_model",call_model)
builder.add_node("tools",tool_node)
builder.add_edge(START,"call_model")
builder.add_conditional_edges("call_model",should_continue,["tools",END])
builder.add_edge("tools","call_model")


