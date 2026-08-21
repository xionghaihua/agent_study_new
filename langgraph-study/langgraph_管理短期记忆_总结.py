#修剪或删除消息的问题在于，可能会因剔除消息队列而丢失消息，因此，一些应用程序受益于一种更复杂的方法，使用聊天模型来汇总消息历史记录
#pip install langmem

from typing import Annotated,TypedDict,Any
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage,HumanMessage,AIMessage,SystemMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term.summarization import SummarizationNode,RunningSummary
import os
from dotenv import load_dotenv
load_dotenv()
llm = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    #model="openai:qwen3.7-plus",
    model="openai:MiniMax-M2.1",
    temperature=0
)

#创建一个专用于摘要的模型实例，限制输出最多128tokens
summarization_model = llm.bind(max_tokens=128)

#定义状态结构，包含对话历史和摘要上下文
class State(MessagesState):
    context: dict[str,RunningSummary] #用于存储用户摘要记忆

#定义输入格式，传给call_model函数调用
class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str,RunningSummary]
#首次生成摘要
initial_summary_prompt = ChatPromptTemplate.from_template(
    """请阅读以下对话内容，并生成一个简洁的摘要，用于帮助理解对话的主要内容:
    对话内容:
    {messages}
    摘要: """
)
#在已有摘要基础上追加新的对话内容，更新摘要
existing_summary_prompt = ChatPromptTemplate.from_template(
    """你之前已经生成了如下摘要:
    {existing_summary}
    现在，对话继续发展了，请根据新增的对话内容，更新这个摘要，使其覆盖所有关键内容
    
    新增对话内容:
    {messages}
    更新后的摘要: """
)

#用于最终调用模型之前，将摘要和剩余消息一起传入模型
final_prompt = ChatPromptTemplate.from_template(
    """你是一位智能助理，以下是用户和你的对话摘要，可帮助你快速理解上下文:
    摘要:
    {summary}
    
    这是对话中未被总结的新消息，请继续处理这些信息:
    {messages}
    """
)
#创建摘要节点，超过一定token数对历史消息自动进行摘要
summarization_node = SummarizationNode(
    token_counter = count_tokens_approximately,  #使用近似token计算
    model = summarization_model,
    max_tokens = 200, #进行摘要之前，传给模型的输入上下文的最大token长度限制
    max_tokens_before_summary = 50, #超过这个数就会触发摘要
    max_summary_tokens = 128, #每次摘要最多保留128 tokens
    initial_summary_prompt = initial_summary_prompt,
    existing_summary_prompt = existing_summary_prompt,
    final_prompt = final_prompt,
)

#模型调用节点
def call_llm(state:LLMInputState):
    response = llm.invoke(state["summarized_messages"])
    for i,msg in enumerate(state["summarized_messages"]):
        print(f"[{i}] {msg.type}:{msg.content}....")
    print(f"messages:{[response]}")
    print(f"context: {state.get('context',{})}")
    print("=" * 70 )
    return {
        "messages": [response],
        "context": state.get("context",{}), #把上下文原样返回，里面就有摘要
    }

checkpointer = InMemorySaver()
builder = StateGraph(State)

builder.add_node("call_llm", call_llm)
builder.add_node("summarize", summarization_node)
builder.add_edge(START,"summarize")
builder.add_edge("summarize","call_llm")

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable":{"thread_id":"1"}}
graph.invoke({"messages":[{"role":"user","content":"我叫初见，是一名大模型开发"}]},config=config)

graph.invoke({"messages":[{"role":"user","content":"请写一首关于猫的诗"}]},config=config)
graph.invoke({"messages":[{"role":"user","content":"现在对狗做一样的事情"}]},config=config)
final_response = graph.invoke({"messages":[{"role":"user","content":"你还记得我的名字吗？"}]},config=config)

final_response["messages"][-1].pretty_print()
print("\n摘要记忆内容(summary):",final_response)
