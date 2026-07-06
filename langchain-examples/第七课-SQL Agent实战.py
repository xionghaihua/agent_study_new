"""
langchain核心应用--与数据库交互
知识点：
- SQLDatabase组件
- SQLDatabaseToolkit
- 数据库连接与安全
- Human-in-the-loop审核
- 真实的SQL Agent实现
"""
from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import  SQLDatabaseToolkit
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()
langfuse_handler = CallbackHandler()

def create_sample_db(db_path: str = "./lesson07_sample.db"):
    """创建示例SQlite数据库"""
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()  #用于执行SQL命令并获取结果的指针
    #创建用户表
    cursor.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT not NULL,
            age INTEGER,
            city TEXT,
            salary REAL
        )
        """
    )
    #创建订单表
    cursor.execute(
        """
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    # execute 一条一条的执行
    #executemany：一次性执行多条结构相同的sql

    users = [
        (1, "张三", 28, '北京', 15000),
        (2, "李四", 25, '上海', 18000),
        (3, "王五", 32, '广州', 16500),
        (4, "赵六", 26, '深圳', 20000),
        (5, "孙七", 30, '杭州', 17200),
        (6, "周八", 27, '成都', 13500),
        (7, "吴九", 35, '南京', 19000),
        (8, "郑十", 29, '武汉', 14800)
    ]
    cursor.executemany("INSERT INTO users VALUES (?,?,?,?,?)", users)

    orders = [
        (1001, 1, "IPhone", 5999),
        (1002, 2, "华为Mate70", 4999),
        (1003, 3, "小米15", 3999),
        (1004, 1, "MacBook Pro", 9999),
        (1005, 4, "AirPods", 1299),
        (1006, 5, "平板iPad", 3499),
        (1007, 2, "机械键盘", 399),
        (1008, 6, "蓝牙耳机", 299),
        (1009, 7, "手表Watch", 2199),
        (1010, 8, "移动硬盘", 599)
    ]
    cursor.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    conn.commit()
    conn.close()
    print(f"示例数据库已创建：{db_path}")

def example_1():
    print("\n====示例1:真实SQLDatabase与基础查询")
    #创建示例数据库
    create_sample_db()
    """
    SQLDatabase组件：
    是Langchain提供的数据库抽象层，封装了SQLAlchemy引擎，提供安全的SQL安全接口
    支持的数据库：
    - SQLLite
    - PostgreSQL
    - MySQL
    - Oracle
    核心功能：
    get_table_names() 列出所有表
    get_table_info() 查看表结构（DDL语句）
    run(sql): 执行SQL查询并返回结果
    get_context()   获取数据库上下文信息，用于Agent
    """

    #初始化SQLDatabase
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")

    #查看数据库结构
    print("\n====数据库结构信息=====")
    print(f"可用表：{db.get_usable_table_names()}")
    print(f"\n表结构(table_info): {db.get_table_info()}")
    #测试直接查询
    result = db.run("SELECT * FROM users LIMIT 2")
    print(f"查询结果:\n{result}")
    """
    使用SQLDatabaseToolKit创建Agent
    SQLDatabaseToolkit是Langchain提供的SQL工具包，他会自动生成多个工具供Agent使用
    - sql_db_query 执行SQL查询
    - sql_db_schema 查询表结构
    - sql_db_list_tables 列出所有表
    - sql_db_query_checker 检查SQL语法
    Agent会自动决定使用哪个工具
    """
    model=init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    #创建SQL工具包
    toolkit=SQLDatabaseToolkit(db=db,llm=model)
    tools = toolkit.get_tools()
    print(f"\n====Toolkit提供的工具===========")
    for t in tools:
        print(f"===={t.name}:{t.description}")
    # 创建SQL Agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""你是一个数据库专家助手，帮助用户查询数据
        请使用提供的工具来搜索数据库结构并执行查询，如果查询失败，请分析错误原因并尝试修正
        """
    )
    print("\n====测试1，查询所有用户====")

    result1 = agent.invoke(
        {"messages":[("user","列出所有用户的姓名和城市")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n=====测试2:条件查询======")
    result2 = agent.invoke(
        {"messages":[("user","查询心脏超过15000的用户")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI:{result2['messages'][-1].content}")

def example_2():
    """
    示例2: 手动封装SQL工具与权限控制
    目标：掌握如何自定义SQL工具
    知识点：
    - 手动创建查询工具
    - SQL注入防护
    - 只读访问控制
    - 查询结果格式化
    """
    print("\n=======示例2:手动封装SQL工具与权限控制======")
    #1.初始化数据库
    create_sample_db()
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")
    """
    通常生产环境，我们通常需要
    1.限制查询类型，只允许SELECT，禁止INSERT,UPDATE,DELETE
    2.限制访问表，只允许访问特定表
    3.限制返回行数，防止全表扫描拖慢系统
    4.自定义错误处理，友好的错误提示
    5.查询日志，记录所有执行的SQL
    """
    @tool
    def safe_sql_query(sql:str)->str:
        """
        安全执行SQL查询（只读模式）
        参数：
          sql: SQL查询语句
        返回：
            查询结果，最多返回10行
        """
        #安全检查，只允许SELECT
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            return "错误：只允许执行SELECT查询"
        #安全检查，阻止危险关键字
        dangerous = ["DROP","DELETE","UPDATE","INSERT","ALTER","CREATE","TRUNCATE"]
        for keyword in dangerous:
            if keyword in sql_stripped:
                return f"错误：禁止执行{keyword}操作"
        try:
            #限制返回行数
            if "LIMIT" not in sql.upper():
               sql = sql.rstrip(";") + "LIMIT 10"
            result = db.run(sql)
            if not result or result.strip()=="":
                return "查询结构为空"
            return f"查询成功:\n{result}"
        except Exception as e:
            return f"查询执行失败：{str(e)}"

    model=init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model,
        tools=[safe_sql_query],
        system_prompt="""你是一个数据库助手，帮助用户安全的查询数据
        - 只执行SELECT查询
        - 将SQL结果结构好为易读的格式
        - 如果查询失败，解释错误原因
        """
    )
    print("\n=====测试1:正常查询=======")
    result1 = agent.invoke(
        {"messages":[("user","查询users表中所有用户的姓名和薪资")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n=====测试：尝试危险操作======")
    result2 = agent.invoke(
        {"messages":[("user","删除user表中id为1的数据")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI:{result2['messages'][-1].content}")

def example_3():
    """
    数据分析Agent
    目标：创建能进行数据分析的SQL Agent
    - SQL聚合查询
    - 多表JOIN查询
    - 统计分析
    :return:
    """
    print("\n=======示例3:数据分析Agent=========")
    #初始化数据库
    create_sample_db()
    db = SQLDatabase.from_uri("sqlite:///./lesson07_sample.db")
    @tool
    def analyze_data(query_type:str) ->str:
        """
        执行数据分析查询
        参数：
            query_type:分析类型
            - “user_stats” 用户统计，平均薪资，最高薪资等
            - “order_stats" 订单统计，总订单数，总金额等
            - “user_orders” 用户订单分析，JOIN查询
        返回：
            分析结果
        """
        try:
            if query_type == "user_stats":
                sql="""
                    SELECT
                        COUNT(*) as 用户总数,
                        ROUND(AVG(salary),0) as 平均薪资,
                        MAX(salary) as 最高薪资,
                        MIN(salary) as 最低薪资,
                        ROUND(AVG(age),1) as 平均年龄
                    FROM users
                """
                result = db.run(sql)
                return f"用户统计分析:\n{result}"
            elif query_type == "order_stats":
                sql="""
                    SELECT
                       COUNT(*) as 订单总数,
                       SUM(amount) as 订单总金额,
                       ROUND(AVG(amount),0) as 平均订单金额,
                       MAX(amount) as 最高订单金额
                    FROM orders
                """
                result = db.run(sql)
                return f"订单统计分析:\n{result}"
            elif query_type == "user_orders":
                sql="""
                    SELECT
                        u.name as 用户姓名,
                        COUNT(o.order_id) as 订单数,
                        SUM(o.amount as 消费金额
                    FROM users u LEFT JOIN orders o ON u.id = o.user_id
                    GROUP BY u.name
                    ORDER BY 消费总额 DESC
                """
                result = db.run(sql)
                return f"用户订单分析:\n{result}"
            else:
                return f"未知的分析类型:{query_type}"
        except Exception as e:
            return f"查询执行失败:{str(e)}"

    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent = create_agent(
        model=model,
        tools=[analyze_data],
        system_prompt="""你是一个数据分析专家，擅长
        -用户数据分析
        -订单数据分析
        -用户消费行为分析
        请根据数据给出专业的分析和建议"""
    )
    print("\n=====测试1:用户统计=====")
    result1 = agent.invoke(
        {"messages":[("user","帮我统计一下用户的基本数据")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n======测试2:用户消费分析======")
    result2 = agent.invoke(
        {"messages":[("user","分析一下用户的订单和消费情况")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI:{result2['messages'][-1].content}")

def example_4():
    """
    使用langgraph interrupt机制实现人工审核
    - interrupt_before机制
    - 暂停/继续执行流程
    - 人工审核决策
    - 审核拒绝时返回AIMessage
    :return:
    """
    print("\n=========示例4:Human-in-the-loop审核======")
    create_sample_db()
    db = SQLDatabase.from_uri("sqlite:///lesson07_sample.db")
    @tool
    def execute_sql(sql:str)->str:
        """执行SQL查询，只允许SELECT"""
        if not sql.strip().upper().startswith("SELECT"):
            return "错误:只允许执行SELECT查询"
        try:
            result = db.run(sql)
            if not result or result.strip()=="":
                return "查询结果为空"
            return f"查询成功:\n{result}"
        except Exception as e:
            return f"查询失败:{str(e)}"

    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent = create_agent(
        model=model,
        tools=[execute_sql],
        checkpointer=MemorySaver(),
        interrupt_before=["tools"]
    )
    print("\n=====场景1:简单查询=====")
    config1 = {
        "configurable":{"thread_id":"sql_review_1"},
        "callbacks":[langfuse_handler],
    }
    result1 = agent.invoke(
        {"messages":[("user","查询所有用户的姓名和城市")]},
        config=config1,
    )
    ai_msg = result1["messages"][-1]
    if ai_msg.tool_calls:
        sql_to_exec = ai_msg.tool_calls[0]["args"]["sql"]
        print(f"Agent已暂停，它准备执行以下SQL")
        print(f"SQL:{sql_to_exec}")
        confirm = input("\n是否批准执行?(y/n):")
        if confirm.lower()=="y":
            print("审核通过，继续执行....")
            final_result = agent.invoke(None,config=config1)
            print(f"最终结果:{final_result['messages'][-1].content}")
        else:
            print("审核拒绝，操作取消")
            reject_msg = AIMessage(
                content="您的查询需要人工审核，但已被管理员拒绝，联系管理员"
            )
            agent.update_state(config1,{"messages":[reject_msg]})
            updated_state = agent.get_state(config1)
            print(f"AI回复：{updated_state.values['messages'][-1].content}")
    else:
        print("\nAgent没有调用工具，直接回复")
        print(ai_msg.content)




def main(example_number:int):
    print("="*60)
    print("第7课-SQL Agent实战")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(4)




