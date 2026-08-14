#pip install -U langgraph-checkpoint-redis
import uuid
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.redis import RedisSaver #短期
from langgraph.store.redis import RedisStore #长期
from langgraph.runtime import  Runtime
from dataclasses import dataclass
from langchain.messages import HumanMessage,AIMessage,SystemMessage
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

#定义节点
def call_model(state:MessagesState, runtime:Runtime[Context]):
    user_id = runtime.context.user_id
    #长期记忆，创建命名空间
    namespace = ("memories",user_id)
    #获取用户最新的消息
    last_message = state["messages"][-1].content

    #长期记忆，通过runtime进行搜索,得到当前用户相关的,search是一个模糊查询，get是精确查询
    memories = runtime.store.search(namespace,query=last_message)

    #将检索到的长期记忆填充到提示词中
    user_info= "\n".join([d.value["data"] for d in memories])
    system_prompt = (f"你是一个乐于助人的助手，请根据已知用户信息进行回复:"
                    f"用户信息:{user_info}"
                    f"如果用户信息不为空才根据已知的用户信息进行回答")
    #调用模型
    result = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])

    #怎么存入长期记忆，长期记忆，一般存储用户相关的信息，比如用户的习惯等
    if "记住" in last_message:
        #简单提取记住后面的内容（实际生成可用LLM提取）
        memory_content = last_message.replace("记住","").strip(": : ")
        #key:str(uuid.uuid4()),要求唯一
        runtime.store.put(namespace,str(uuid.uuid4()),{"data":memory_content})
        print(f"----[系统日志]已存入长期记忆：{memory_content}")

    return {"messages": result}

#构建图
DB_URL = "redis://:Aa123456@172.16.181.128:6380/0"
with RedisStore.from_conn_string(DB_URL) as store, \
    RedisSaver.from_conn_string(DB_URL) as checkpointer:
    #初始化,连接对象
    store.setup()
    checkpointer.setup()
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
                "user_id":current_user_id,
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

