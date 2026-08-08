#-*- coding: utf-8 -*-
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langfuse.langchain import CallbackHandler
import os

load_dotenv()
langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.6-plus",
    temperature=0.2
)

search_tool = TavilySearch(max_results=2)
tools = [search_tool]
llm_with_tools = llm.bind_tools(tools)

def chat_node(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("chat", chat_node)
workflow.add_node("tools", ToolNode(tools=tools))
workflow.add_conditional_edges("chat", tools_condition)
workflow.add_edge("tools", "chat")
workflow.set_entry_point("chat")

memory = MemorySaver()
#编译工作流，传入memory开启记忆功能
app = workflow.compile(checkpointer=memory)

def chat_with_memory():
    print("智能对话机器人（支持记忆和搜索）")
    print("输入exit退出对话")
    thread_id = "user_session_1"
    # callbacks 放在graph config，全局生效
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler]
    }
    while True:
        try:
            user_input = input("\n您：").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", '退出', "q"]:
                print("机器人：再见")
                break
            result = app.invoke(
                {"messages": [("user", user_input)]},
                config=config
            )
            if result["messages"]:
                last_message = result["messages"][-1]
                print("机器人：", last_message.content)
        except KeyboardInterrupt:
            print("\n对话结束")
            break
        except Exception as e:
            print(f"出错：{str(e)}")

if __name__ == "__main__":
    chat_with_memory()