#pip install -U "psycopg[binary,pool]" langgraph-checkpoint-postgres

import uuid
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.runtime import Runtime
load_dotenv()

#初始化模型

llm = init_chat_model(
    base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="openai:qwen3.7-plus",
    temperature=0
)

@dataclass
class Context:
    user_id: str


#定义节点逻辑
def call_model(
        state:MessagesState,
        runtime:Runtime[Context],
):
    user_id = runtime.context.user_id
    namespace = ("memories",user_id)
    #检查长期记忆
    last_user_msg = state["messages"][-1].content
    memories = runtime.store.search(namespace,query=str(last_user_msg))
    info = "\n".join([d.value['data'] for d in memories])
    system_msg = f"您是一个乐于助人的助手，已知用户信息:{info}"
    if "记住" in  last_user_msg:
        memory_content = last_user_msg.replace("记住","").strip(": : ")
        runtime.store.put(namespace,str(uuid.uuid4()),{"data":memory_content})
        print(f"----[系统日志]已存入长期记忆：{memory_content}")
    response = llm.invoke(
        [{"role":"system","content":system_msg}] + state["messages"]
    )
    return {"messages": response}

DB_URI = "postgresql://admin:admin123@172.16.181.128:5435/ai_memory?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store,\
        PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    #第一次初始化的时候需要
    checkpointer.setup()
    store.setup()

    #构建图
    builder = StateGraph(MessagesState)
    builder.add_node("call_model",call_model)
    builder.add_edge(START,"call_model")
    graph = builder.compile(checkpointer=checkpointer,store=store)
#交互
    current_thread_id = "1"
    current_user_id = "user_v1"
    print("======langgaph交互系统=======")
    print("指令说明：输入 'switch' 切换会话，'exit' 退出程序")
    while True:
        prompt = f"\n[当前线程:{current_thread_id}] 用户:"
        user_input = input(prompt).strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "switch":
            new_id = input("请输入新的thread_id，如(2,3,4...):")
            current_thread_id = new_id
            print(f"----已切换到现场{current_thread_id}-------")
            continue
        if not user_input:
            continue
        #构建配置
        config = {
            "configurable":{
                "thread_id":current_thread_id,
                "user_id":current_user_id
            }
        }
        #执行流式输出（使用messages模式）
        for chunk in graph.stream(
                {"messages":[{"role":"user","content":user_input}]},
            config=config,
            stream_mode="messages",
            version="v2",
            context=Context(user_id=current_user_id),
        ):
            if chunk["type"] == "messages":
                result,metadata = chunk["data"]
                print(result.content,end="",flush=True)
