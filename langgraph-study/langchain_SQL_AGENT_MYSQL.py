#pip install pymysql DBUtils

import os
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dbutils.pooled_db import PooledDB
import pymysql
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()
load_dotenv()

#mysql 连接池
MYSQL_POOL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "172.16.181.128"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB"),
    "charset": "utf8mb4",
    "connect_timeout": 10,
}
# 连接池全局单例
POOL = PooledDB(
    creator=pymysql,
    maxconnections=10,       # 最大连接数
    mincached=2,             # 初始化空闲连接
    maxcached=5,             # 最大空闲连接
    maxshared=3,
    blocking=True,           # 连接耗尽时阻塞等待
    maxusage=None,
    setsession=[],
    ping=1,                  # 获取连接前ping检测有效性
    **MYSQL_POOL_CONFIG
)
def get_pool_connection():
    """从连接池获取连接"""
    return POOL.connection()
# ===================== 数据库工具 =====================
@tool
def list_tables()->str:
    """列出数据库中所有表"""
    conn = get_pool_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            rows = cur.fetchall()
            tables = [row[0] for row in rows]
        return f"数据库中表:{','.join(tables)}"
    finally:
        conn.close() #归还连接到池，不是真正关闭
@tool
def get_table_schema(table_name:str)->str:
    """查询数据库表结构信息"""
    conn = get_pool_connection()
    try:
        with conn.cursor() as cur:
            sql = """
            SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION;
            """
            cur.execute(sql,(MYSQL_POOL_CONFIG["database"], table_name))
            columns = cur.fetchall()
            if not columns:
                return f"表'{table_name}'不存在"
            info = []
            for col,dtype,comment in columns:
                info.append(f"- {col}: {dtype} 【备注：{comment}】")
            return f"表`{table_name}`结构：\n" + "\n".join(info)
    finally:
        conn.close()
@tool
def execute_sql_query(sql: str) -> str:
    """
    执行只读SQL查询，仅允许SELECT语句
    安全限制：禁止UPDATE/DELETE/INSERT/DROP/ALTER等
    """
    sql_stripped = sql.strip().upper()
    sql_lower = sql_stripped.lower()
    # 安全校验
    if not sql_stripped.startswith("SELECT"):
        return "【安全拦截】仅允许执行SELECT只读查询，禁止修改类SQL"
    blacklist = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "load"]
    for keyword in blacklist:
        if keyword in sql_lower:
            return f"【安全拦截】禁止包含关键字：{keyword}"

    # 增加最大执行时间5秒，防止慢查询占用连接
    if "limit" not in sql_lower:
        sql += " LIMIT 100;"
    conn = get_pool_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SET MAX_EXECUTION_TIME=5000;")
            cur.execute(sql)
            result = cur.fetchall()
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"【SQL执行异常】{str(e)}"
    finally:
        conn.close()
# ===================== Agent 启动 =====================
def example_1():
    print("\n====示例1:MySQL SQL Agent（连接池版本）=========")
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    sql_agent = create_agent(
        model=model,
        tools=[list_tables, get_table_schema, execute_sql_query],
        checkpointer=InMemorySaver(),
        system_prompt="""你是MySQL数据分析助手。
    工作流程：
    1. 先用list_tables查看数据库有哪些表
    2. 使用get_table_schema查看目标表字段结构
    3. 根据用户问题构造标准SELECT语句，调用execute_sql_query查询
    约束：
    - 只能生成SELECT语句，严禁任何增删改操作
    - 查询尽量带上合理LIMIT，避免一次性返回超大结果集
    - 复杂统计优先使用COUNT/SUM等聚合函数
    - 返回结果后，使用自然语言总结数据，不要直接甩原始JSON
    """,
    )

    config = {"configurable": {"thread_id": "mysql-session-001"},"callbacks":[langfuse_handler]}
    print("\n=======测试1：查询表列表=======")
    res1 = sql_agent.invoke(
        {"messages": [("user", "数据库里有哪些表？")]},
        config=config
    )
    print(f"AI：{res1['messages'][-1].content}")

    print("\n=======测试2：查看表结构=======")
    res2 = sql_agent.invoke(
        {"messages": [("user", "查看orders表结构")]},
        config=config
    )
    print(f"AI：{res2['messages'][-1].content}")

    print("\n=======测试3：带条件查询数据=======")
    res3 = sql_agent.invoke(
        {"messages": [("user", "查询user_id=1001的所有订单")]},
        config=config
    )
    print(f"AI：{res3['messages'][-1].content}")


if __name__ == "__main__":
    example_1()