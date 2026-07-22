#使用postgresql添加短期记忆
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os
print("======示例:使用postgresql持久化短期记忆=============")
# 短期记忆：PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
DB_URI = "postgresql://admin:admin123@172.16.181.128:5435/ai_memory"

load_dotenv()

try:
    # 创建连接池，生产环境推荐
    # 如果本地单线程，也可以直接用psycopg.connect()
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }
    pool = ConnectionPool(
        conninfo=DB_URI,
        max_size=5,  # 连接池最大连接数，限制兵法请求数量
        kwargs=connection_kwargs
    )
    # 初始化postgresSaver
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # 首次运行必需调用，创建checkpoint表结构
    """
    setup()做了啥？
    在postgresql创建checkpoint相关表
    - checkpoints 存储每个thread检查点
    - checkpoint_writes 存储待写入的更新
    - checkpoint_blob 存储大对象
    """
    tools = [TavilySearch(max_size=1)]
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个聊天助手，记住我们聊过的内容",
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "pg_1"}}
    query1 = "请问现任的美国总统是谁？他的年龄是多少？请用中文回答"
    query2 = "请问我上一个问题问了什么？"
    result1 = agent.invoke({"messages":[{"role":"user","content":query1}]},config=config)
    print(result1["messages"][-1].content)
    result2 = agent.invoke({"messages":[{"role":"user","content":query2}]},config=config)
    print(result2["messages"][-1].content)
except Exception as e:
    print(f"连接数据库失败：{e}")