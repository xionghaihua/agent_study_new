"""
目标：通过实战项目掌握langgraph的高级应用
示例1: 自定义SQL Agent
示例2: 复杂工作流编排
示例3: 错误处理与容错
示例4: 基于Milvus的RAG Agent
示例5: RAG+Agent综合
知识点:
- 自定义SQL Agent
- 复杂工作流编排
- 错误处理与容错
- 自定义RAG Agent
- 生产级实现模式
"""
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

from sqlalchemy.testing.suite.test_reflection import metadata

load_dotenv()

langfuse_handler = CallbackHandler()

#工具函数：初始化嵌入模型

def init_embedding_model(method:str="dashscope"):
    """
    初始化嵌入模型
    参数：
        method: 初始化方法，可选值
        - "dashscope",阿里云的DashScope API
        - "openai": OpenAI 官方API
        - “init_embeddings": langchain通用init_embeddings函数
        - "ollama" : 本地Ollama服务
    返回:
        嵌入模型实例

    案例:
        embedding = init_embedding_model("dashscope")
    """
    if method == "dashscope":
        from langchain_community.embeddings import DatabricksEmbeddings
        return DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        )
    elif method == "openai":
        from langchain_openai.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif method == "init_embeddings":
        from langchain.embeddings import init_embeddings
        return init_embeddings(
            model="text-embedding-v1",
            provider="openai",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            check_embedding_ctx_length = False,
        )
    else:
        raise ValueError(f"不支持的嵌入模型初始化方法: {method}")

#示例1: 自定义SQL Agent 基础篇

def example_1():
    """
    示例1: 自定义SQL Agent
    目标: 构建能执行SQL查询的数据分析Agent
    知识点:
    - SQL 执行工具
    - 表结构查询
    - 安全限制（只读查询）
    """
    print("\n====示例1:自定义SQL Agent=========")
    from langchain.agents import create_agent
    from langchain.tools import tool
    import json
    #模拟数据库表结构
    database_schema = {
        "users":{
            "columns":{
                "user_id": "INT,PRIMARY KEY",
                "name": "VARCHAR(50)",
                "email": "VARCHAR(60)",
                "city": "VARCHAR(60)",
                "level": "VARCHAR(20)",
            },
            "data":[
                {"user_id":1001,"name":"李娜","email":"lina@163.com","city":"北京","level":"VIP"},
                {"user_id": 1002, "name": "王冰", "email": "wangbing@163.com", "city": "广州", "level": "普通会员"},
                {"user_id": 1003, "name": "吴文斌", "email": "wuwenbin@163.com", "city": "武汉", "level": "钻石会员"},
                {"user_id": 1004, "name": "熊伊", "email": "xiongyi@163.com", "city": "深圳", "level": "黄金会员"},
            ],
        },
        "orders":{
            "columns":{
                "order_id": "INT,PRIMARY KEY",
                "user_id": "INT",
                "product": "VARCHAR(100)",
                "amount": "DECIMAL(10,2)",
                "order_date": "DATE",
                "status": "VARCHAR(20)",
            },
            "data":[
                {"order_id":1001,"user_id":1001,"product":"iphone 15","amount":7999,"order_date":"2024-01-10","status":"已完成"},
                {"order_id": 1002, "user_id": 1002, "product": "AirPods Pro", "amount": 1999, "order_date": "2024-02-15",
                 "status": "已完成"},
                {"order_id": 1003, "user_id": 1001, "product": "华为手机", "amount": 3999, "order_date": "2025-02-10",
                 "status": "已完成"},
                {"order_id": 1004, "user_id": 1003, "product": "小米电脑", "amount": 4999, "order_date": "2025-05-20",
                 "status": "已完成"},
            ],
        },
    }

    @tool
    def get_table_schema(table_name:str)->str:
        """查询数据库表的结构信息"""
        table = database_schema.get(table_name)
        if not table:
            return f"表'{table_name}'不存在，可用的表有:{','.join(list(database_schema.keys()))}"
        columns_info = "\n".join(f"-{col}:{dtype}" for col,dtype in table["columns"].items())
        return f"表‘{table_name}'的表结构:\n{columns_info}"

    @tool
    def list_tables()->str:
        """列出数据库中所有表"""
        return f"数据库中表：{','.join(database_schema.keys())}"

    @tool
    def execute_sql_query(sql:str)->str:
        """
        执行只读SQL查询，仅仅支持SELECT
        支持特性：SELECT * / 指定字段、FROM、WHERE等值过滤、COUNT、LIMIT
        暂不支持 JOIN、GROUP BY、复杂运算符(> < LIKE)、子查询
        """
        sql_stripped = sql.strip().upper()
        sql_origin = sql.strip()
        sql_lower = sql_stripped.lower()
        #安全校验，近允许SELECT
        if not sql_stripped.startswith("SELECT"):
            return "错误：仅允许执行SELECT查询"
        #禁止多表JOIN
        if "join" in sql_lower:
            return "错误:当前模拟数据库不支持JOIN多表查询"

        #解析FROM获取表名
        from_idx = sql_lower.find("from ")
        if from_idx == -1:
            return "错误：SQL语法错误，未找到FROM和表名"
        table_name = sql_lower[from_idx + 5:].strip().split()[0].strip(";")
        table = database_schema.get(table_name)
        if not table:
            return f"错误：表{table_name}不存在"
        sources_data = table["data"]
        filter_data = sources_data.copy()
        where_idx = sql_lower.find("where ")
        limit_num = None
        if where_idx != -1:
            #获取WHERE之后的语句，截断LIMIT
            where_clause = sql_lower[where_idx + 6:]
            limit_pos = where_clause.find("limit")
            if limit_pos != -1:
                where_clause = where_clause[:limit_pos].strip()
            if "=" in where_clause:
                left,right = where_clause.split("=",1)
                col_name = left.strip()
                val_raw = right.strip().strip(";'\"")
                # 自动识别数字/字符串
                try:
                    val_target = int(val_raw)
                except ValueError:
                    try:
                        val_target = float(val_raw)
                    except ValueError:
                        val_target = val_raw
                # 执行过滤
                filter_data = [row for row in filter_data if row.get(col_name) == val_target]
        # 解析 LIMIT
        limit_idx = sql_lower.find("limit")
        if limit_idx != -1:
            limit_str = sql_lower[limit_idx + 5:].strip().split()[0].strip(";")
            if limit_str.isdigit():
                limit_num = int(limit_str)
                filter_data = filter_data[:limit_num]

        # COUNT 统计
        if "count(" in sql_lower:
            return json.dumps({"count": len(filter_data)}, ensure_ascii=False)

        # 简易字段筛选逻辑 SELECT col1,col2 FROM xxx
        select_part = sql_lower[len("select"):sql_lower.find("from")].strip()
        if select_part != "*":
            select_cols = [c.strip() for c in select_part.split(",")]
            temp_result = []
            for row in filter_data:
                new_row = {k: v for k, v in row.items() if k in select_cols}
                temp_result.append(new_row)
            filter_data = temp_result

        return json.dumps(filter_data, ensure_ascii=False, indent=2)




    #初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    #创建SQL Agent
    sql_agent = create_agent(
        model=model,
        tools=[list_tables,get_table_schema,execute_sql_query],
        checkpointer=InMemorySaver(),
        system_prompt="""你是一个数据库查询助手。
工作流程：
1. 先用list_tables查看有哪些表
2. 用get_table_schema了解目标表结构
3. 构造合法SELECT语句，调用execute_sql_query执行查询
约束：
- 只能生成SELECT语句，禁止UPDATE/DELETE/INSERT/DROP
- 当前数据库不支持JOIN多表关联
- 需要筛选条件时使用WHERE col='xxx'等值查询
- 用户只需要汇总结论时，优先使用COUNT统计
""",
    )
    #测试
    print("\n=======测试1:查询orders表数据===========")
    config = {"configurable":{"thread_id":"sql-session-001"},"callbacks":[langfuse_handler]}
    result = sql_agent.invoke(
        {"messages":[("user","orders表有哪些数据？帮我查询所有记录")]},
        config=config,
    )
    print(f"用户:orders表有哪些数据?")
    print(f"AI:{result['messages'][-1].content}")
    print("\n=======测试2:带WHERE条件查询（用户id=1001订单）===========")
    result2 = sql_agent.invoke(
        {"messages": [("user", "查询user_id等于1001的所有订单")]},
        config=config,
    )
    print(f"用户：查询user_id等于1001的所有订单")
    print(f"AI：{result2['messages'][-1].content}")

    print("\n=======测试3:COUNT统计===========")
    result3 = sql_agent.invoke(
        {"messages": [("user", "统计北京的用户数量")]},
        config=config,
    )
    print(f"用户：统计北京的用户数量")
    print(f"AI：{result3['messages'][-1].content}")
    print(f"用户：查询user_id等于1001的所有订单")
    print(f"AI：{result2['messages'][-1].content}")

    print("\n=======测试3:COUNT统计===========")
    result3 = sql_agent.invoke(
        {"messages": [("user", "统计北京的用户数量")]},
        config=config,
    )
    print(f"用户：统计北京的用户数量")
    print(f"AI：{result3['messages'][-1].content}")


def example_2():
    """
    示例2: 复杂工作流编排
    目标： 使用langgraph编排多步骤的复杂工作流
    知识点：
    - 多节点工作流
    - 条件分支
    - 状态传递
    """
    print("\n========示例2:复杂工作流编排===========")
    from langchain.agents import create_agent
    from langchain.tools import tool
    @tool
    def classify_query(query:str)->str:
        """对用户查询进行分类"""
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["怎么","如何","配置","安装","代码","报错"]):
            return "technical"
        elif any(kw in query_lower for kw in ["价格","购买","优惠","推荐","比较"]):
            return "product"
        elif any(kw in query_lower for kw in ["退货","退款","维修","投诉","订单"]):
            return "support"
        return "general"

    @tool
    def get_technical_answer(query:str)->str:
        """获取技术问题的答案"""
        tech_db = {
            "安装":"安装步骤：1)下载安装包；2）进行安装程序；3）完成配置；4）验证安装。",
            "配置":"配置指南:请参考官网文档的配置章节。",
            "报错":"错误排除:1)查看日志；2）检查配置；3）确认依赖版本;"
        }
        for keyword,answer in tech_db.items():
            if keyword in query:
                return answer
        return "暂未找到相关技术依赖"
    @tool
    def get_product_info(query:str)->str:
        """获取产品信息"""
        product_db = {
            "基础版": {"name":"基础版","price":99,"features":["核心功能","基础支持"]},
            "专业版": {"name": "专业版", "price": 299, "features": ["全部功能", "优先支持"]},
            "企业版": {"name": "企业版", "price": 999, "features": ["定制功能", "专属支持"]},
        }
        results = []
        for name,info in product_db.items():
            if name in query:
                results.append(f"{info['name']}:¥{info['price']}月,功能:{",".join(info['features'])}")
        return "\n\n".join(results) if results else "暂无相关产品"
    @tool
    def create_support_ticket(query:str,priority:str="medium")->str:
        """创建售后工单"""
        import random
        ticket_id = f"TK-{random.randint(10000,99999)}"
        eta_map = {"low":"48小时","medium":"24小时","high":"4小时"}
        return f"工单已创建,编号:{ticket_id}，预计处理时间:{eta_map.get(priority,'24小时')}"
    #初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    #创建工作流
    workflow_agent = create_agent(
        model=model,
        tools=[classify_query,get_technical_answer,get_product_info,create_support_ticket],
        checkpointer=InMemorySaver(),
        system_prompt="""
        你是一个智能客服助手,
        工作流程:
        1、使用classify_query对用户查询进行分类
        2、根据分类结果选择对应的工具
            - technical -> get_technical_answer
            - product -> get_product_info
            - support -> create_support_ticket.
        """
    )
    #测试1
    print("\n=====测试1:产品资源=======")
    config={"configurable":{"thread_id":"workflow-session-001"},"callbacks":[langfuse_handler]}
    result = workflow_agent.invoke(
        {"messages":[("user","专业版有什么功能，价格多少")]},
        config=config,
    )
    print(f"AI:{result['messages'][-1].content}")
    


def example_3():
    """
    示例3:错误处理和容错
    目标:掌握langgraph中错误处理最佳实践
    知识点：
    - 工具内部错误处理
    - Agent级别错误恢复
    - 降级策略
    """
    print("\n============示例3:错误处理与容错===========")
    from langchain.agents import create_agent
    from langchain.tools import tool
    import random

    @tool
    def fetch_external_api(endpoint:str)->str:
        """
        调用外部API获取数据
        Args:
            endpoint: 只能选以下其中一个值：weather / news / stock
        """
        api_responses = {
            "weather":{"city":"北京","temperature":25,"condition":"晴朗"},
            "news":{"headlines":["AI技术突破","科技新闻摘要"]},
            "stock":{"symbol":"AAPL","price":178.5,"change":"+1.2%"},
        }
        r = random.random()
        print(r)
        if r <0.3:
            return f"外部API暂时不可用,Endpoint:{endpoint}"
        data = api_responses.get(endpoint)
        print(data)
        if not data:
            return f"未找到端点'{endpoint}‘的数据。"
        import json
        return json.dumps(data,ensure_ascii=False,indent=2)
    @tool
    def fallback_cache(endpoint:str)->str:
        """
        从本地缓存读取数据，降级方案
          Args:
            endpoint: 只能选以下其中一个值：weather / news / stock
        """
        cache_db = {
            "weather": {"city": "北京", "temperature": 24, "condition": "多云"},
            "news": {"headlines": ["昨日新闻摘要"]},
            "stock": {"symbol": "AAPL", "price": 178.5, "change": "+0.7%"},
        }
        data = cache_db.get(endpoint)
        print(data)
        if data:
            import json
            return f"[来自缓存] {json.dumps(data,ensure_ascii=False,indent=2)}"
        return f"缓存中暂无'{endpoint}'数据"

    #初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    #创建带错误处理的Agent
    error_handling_agent = create_agent(
        model=model,
        tools=[fetch_external_api,fallback_cache],
        checkpointer=InMemorySaver(),
        system_prompt="""你是一个具备容错能力的数据助手,
        错误处理策略:
        1.先尝试调用外部API
        2.如果API失败，自动切换到缓存
        请按照这个策略帮助用户获取数据""",
    )
    print("\n=====测试：获取天气数据===========")
    config = {"configurable":{"thread_id":"error-session-001"},"callbacks":[langfuse_handler]}
    result = error_handling_agent.invoke(
        {"messages":[("user","帮我获取天气数据")]},
        config=config,
    )
    print(f"用户:获取天气数据")
    print(f"AI:{result['messages'][-1].content}")


def example_4():
    """
    示例4: 基于Milvus的生产级RAG Agent
    目标：掌握向量存储的基本使用
    知识点:
    - Milvus向量数据库连接
    - 文档存储
    - 相似度搜索
    推荐版本组合:
    langchain-milvus：0.2.2
    pymilvus：2.5.x
    安装命令：
    pip install "langchain-milvus==0.2.2" "pymilvus>=2.5.7,<2.6"
    :return:
    """
    print("\n======示例4:基于Milvus的生产级RAG Agent=======")
    from langchain_milvus import Milvus
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.documents import Document

    #1.初始化嵌入模型
    embedding_model = init_embedding_model(method="dashscope")
    print("嵌入模型初始化成功")
    #2.初始化milvus向量存储
    try:
        #2.初始化Milvus向量存储
        vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri":"http://172.16.181.128:19530"},
            collection_name="rag_agent_kb",
            drop_old=False, #是否丢弃老的数据，生产为False
            auto_id=False,  #是否自动生成ID，生产为False
        )
        print("Milvus向量存储初始化成功")
        test_docs = [
            Document(
                page_content="LangChain是构建LLM应用的框架，提供Agent，工具集成，记忆管理等组件",
                metadata={"source":"langchain"},
            ),
            Document(
                page_content="RAG是检索增强生成技术，流程：检索文档-->生成回答，可以减少幻觉",
                metadata={"source":"rag"},
            ),
            Document(
                page_content="Milvus是开源数据库，支持十亿级向量检索，P99延迟小于30ms",
                metadata={"source":"milvus"},
            ),
        ]

        vector_store.add_documents(test_docs)
        print(f"已添加{len(test_docs)}条文档")
        #4.测试相似度搜索
        results = vector_store.similarity_search("RAG",k=2)
        print(f"检索'RAG',找到{len(results)}条结果:")
        for i,doc in enumerate(results):
            print(f"{i+1}.{doc.page_content[:40]}....")
        #清理
        #vector_store.drop()
    except Exception as e:
        return (f"连接失败:{e}")
        return

    #6.创建RAG Agent（使用模拟知识库)
    knowledge_base_docs = [
        Document(
            page_content="LangChain是一个用于构建大语言模型应用开源框架，它提供了Agent架构，工具集成，记忆管理等核心组件",
            metadata={
                "title":"LangChain框架介绍",
                "source":"https://python.langchain.com/docs/introduction",
                "doc_id": "langchain",
            },
        ),
        Document(
            page_content="RAG是一种结合检索和生成技术，工作流程：用户提问，检索相关文档，将文档作为上下文，生成回答",
            metadata={
                "title": "RAG检索增强生成",
                "source":"https://python.langchain.com/docs/rag",
                "doc_id": "rag",
            },
        ),
        Document(
            page_content="Agent是结合LLM和工具的智能代理系统，核心能力：推理和工具调用",
            metadata={
                "title":"Agent智能代理",
                "source":"https://python.langchain.com/docs/agent",
                "doc_id": "agent",
            }
        ),
        Document(
            page_content="Tool是Agent可以调用的函数，用于执行具体的擦着，使用@tool装饰器，提供清晰的文档字符串",
            metadata={
                "title":"Tool工具定义",
                "source":"https://python.langchain.com/docs/tool",
                "doc_id": "tool",
            }
        ),
    ]
    #模拟知识库
    knowledge_base={
        doc.metadata["doc_id"]: {
            "title":doc.metadata["title"],
            "content":doc.page_content,
            "source":doc.metadata["source"],
        }
        for doc in knowledge_base_docs
    }

    @tool
    def search_knowledge_base(query:str,top_k:int=2)->str:
        """
        在知识库中检索相关文档
        """
        query_lower = query.lower()
        results = []
        for doc_id,doc in knowledge_base.items():
            content_match = any(
                keyword in doc["content"].lower() for keyword in query_lower.split() if len(keyword) > 1
            )
            if content_match:
                results.append(doc)
        results = results[:top_k]
        if not results:
            return f"未在知识库中找到相关内容"
        return "\n\n".join([f"[{r['title']}] \n{r['content']}" for r in results])
    # 初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    #创建RAG Agent
    rag_agent = create_agent(
        model=model,
        tools = [search_knowledge_base],
        checkpointer=InMemorySaver(),
        system_prompt="""你是一个基于知识库的智能助手
        1.先使用search_knowledge_base检索相关文档
        2.基于检索到的内容回答用户问题
        3.如果知识库中没有相关内容，如实告知用户，不能胡编
        """,
    )

    #测试
    print("\n===========测试：知识库问答===========")
    config={"configurable":{"thread_id":"rad-session-001"},"callbacks":[langfuse_handler]}
    result = rag_agent.invoke(
        {"messages":[("user","什么是RAG，它的工作流程是什么?")]},
        config=config,
    )
    print(f"AI:{result['messages'][-1].content}")






def main(example_number:int):
    print("="*60)
    print("第十四章-langgraph实战")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4,

    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(4)