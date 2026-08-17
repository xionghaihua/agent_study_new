import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import pymysql
from dbutils.pooled_db import PooledDB

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sql-agent-server")
# ===================== MySQL 连接池 =====================
MYSQL_POOL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "172.16.181.128"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB"),
    "charset": "utf8mb4",
    "connect_timeout": 10,
}

POOL = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=2,
    maxcached=5,
    maxshared=3,
    blocking=True,
    ping=1,
    **MYSQL_POOL_CONFIG
)
def get_pool_connection():
    return POOL.connection()
# ===================== Tools =====================
@tool
def list_tables() -> str:
    """列出数据库中所有表"""
    conn = get_pool_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            rows = cur.fetchall()
            tables = [row[0] for row in rows]
        return f"数据库中的表：{','.join(tables)}"
    finally:
        conn.close()

@tool
def get_table_schema(table_name: str) -> str:
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
            cur.execute(sql, (MYSQL_POOL_CONFIG["database"], table_name))
            columns = cur.fetchall()
            if not columns:
                return f"表'{table_name}'不存在"
            info = []
            for col, dtype, comment in columns:
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

    if not sql_stripped.startswith("SELECT"):
        return "【安全拦截】仅允许执行SELECT只读查询，禁止修改类SQL"

    blacklist = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "load"]
    for keyword in blacklist:
        if keyword in sql_lower:
            return f"【安全拦截】禁止包含关键字：{keyword}"

    if "limit" not in sql_lower:
        sql += " LIMIT 100;"

    conn = get_pool_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("SET MAX_EXECUTION_TIME=5000;")
            cur.execute(sql)
            result = cur.fetchall()
        logger.info(f"执行SQL: {sql}")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("SQL执行失败")
        return f"【SQL执行异常】{str(e)}"
    finally:
        conn.close()
# ===================== Agent 全局单例 =====================
def build_sql_agent():
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    agent = create_agent(
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
    return agent

SQL_AGENT = build_sql_agent()
# ===================== FastAPI 服务 =====================
app = FastAPI(title="SQL Agent Service", version="1.0")
class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = "default-session"  # 会话ID，支持多轮上下文
class QueryResponse(BaseModel):
    answer: str
    thread_id: str

@app.post("/sql/chat", response_model=QueryResponse)
async def sql_chat(req:QueryRequest):
    logger.info(f"收到请求 thread_id={req.thread_id}, question={req.question}")
    config = {
        "configurable": {"thread_id": req.thread_id}
    }
    try:
        result = SQL_AGENT.invoke(
            {"messages": [("user", req.question)]},
            config=config
        )
        answer = result["messages"][-1].content
        return QueryResponse(answer=answer, thread_id=req.thread_id)
    except Exception as e:
        logger.exception("Agent调用异常")
        raise HTTPException(status_code=500, detail=f"服务异常：{str(e)}")
@app.get("/health")
async def health():
    return {"status": "ok"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_sql_agent_mysql:app", host="0.0.0.0", port=8000, reload=False)