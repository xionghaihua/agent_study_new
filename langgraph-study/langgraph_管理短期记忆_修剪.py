from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()
langfuse_handler = CallbackHandler()
llm = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="openai:qwen3.7-plus",
    temperature=0
)

#不常用
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from langchain_core.messages import HumanMessage

def call_llm(state:MessagesState):
    print(f"修剪前的消息:{state['messages']}")
    messages = trim_messages(
        state['messages'],
        strategy="last",
        token_counter = count_tokens_approximately,
        max_tokens = 100,
        start_on = "human",
        end_on = ("ai","tools","human"),
        allow_partial=True,  # 允许截断对话片段
    )
    print(f"修剪后的消息:{messages}")
    response = llm.invoke(messages)
    return {"messages":response}

checkpointer = InMemorySaver()
builder = StateGraph(MessagesState)
builder.add_node("call_llm",call_llm)
builder.add_edge(START,"call_llm")
graph = builder.compile(checkpointer=checkpointer)

import uuid
thread_id = str(uuid.uuid4())
config = {"configurable":{"thread_id":thread_id},"callbacks":[langfuse_handler]}
graph.invoke({"messages":[{"role":"user","content":"我的名字叫peter"}]},config=config)
graph.invoke({"messages":[{"role":"user","content":"帮我的猫写一首诗"}]},config=config)
graph.invoke({"messages":[{"role":"user","content":"现在对狗做一样的事情"}]},config=config)
final_response = graph.invoke({"messages":[{"role":"user","content":"我的名字叫什么？"}]},config=config)
print(f"最终消息:{final_response['messages']}")
