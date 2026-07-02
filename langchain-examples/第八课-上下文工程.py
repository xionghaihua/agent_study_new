"""
上下文工程实战
=======================
- 静态运行时上下文
- 动态运行时上下文（State)
- 跨对话上下文
- 可变性与生命周期管理
- CheckPointer vs Store 架构对比
- Store API详解
"""
#pip install langgraph-checkpoint-sqlite

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from dataclasses import dataclass
from langgraph.store.sqlite import SqliteStore
from typing import Any,NotRequired
from dotenv import load_dotenv
import os

from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.RuntimeOptimizationRecord import \
    RuntimeOptimizationRecordAddNodesToOptimizeIndices

load_dotenv()
langfuse_handler=CallbackHandler()

def example_1():
    """
    示例1：静态运行上下文
    目标：理解静态上下文的概念和用途
    知识点：
    - 定义上下文：Schema
    - 传递上下文给Agent
    - 工具中访问上下文
    :return:
    """
    print("\n=========示例1:静态运行时上下文===========")
    @dataclass
    class AppContext:
        """应用上下文：包含用户信息和设置"""
        user_id:str
        user_name:str
        language:str="zh"
        timezone:str="Asia/Shanghai"

    #工具：访问上下文

    """
    ToolRuntime[AppContext] ToolRuntime是一个泛类型，其参数指定了runtime.context的具体类型
    runtime.context的类型为AppContext
    
    
    AgentState/Runtime/ToolRuntime对比
    
    AgentState:
        Agent的内存状态，整个Agent执行期间，TypedDict字典；
        messages必需
        structured_response 结构化回复
        使用场景：中间件读写Agent状态
        @before_model
        def mw(state,rt):
            msg=state["messages"]
    Runtime:
        Agent执行时的运行环境，单次invoke
        内置属性：context，model
        使用场景：中间件访问上下文
        @before_model
        def mw(state,rt):
            ctx = rt.context
        
    ToolRuntime:
        工具执行时的运行环境
        内置属性：context，state
        场景：工具访问Agent状态
    """
    @tool
    def get_user_greeting(runtime:ToolRuntime[AppContext])->str:
        """获取用户个性化问候语"""
        ctx = runtime.context
        return f"您好，{ctx.user_name} 您的用户ID为:{ctx.user_id},时区为：{ctx.timezone}"
    @tool
    def get_user_settings(runtime:ToolRuntime[AppContext])->str:
        """获取用户当前设置"""
        print("\n====runtime所有键名和值====")
        print(f"runtime类型：{type(runtime).__name__}")
        for key,value in runtime.__dict__.items():
            print(f"\n{key}: {value}")
        print("================================")
        ctx = runtime.context
        return f"用户设置:\n---语言:{ctx.language}\n---时区：{ctx.timezone}"
    #中间件：追踪状态变化
    from langchain.agents.middleware import AgentState,before_model
    from langgraph.runtime import Runtime

    @before_model
    def print_state(state:AgentState,runtime:Runtime)->dict[str,Any]|None:
        """记录状态变化"""
        print("\n======state所有字段和值=======")
        print(f"state类型：{type(state).__name__}")
        for key,value in state.items():
            print(f"\n{key}: {value}")
        print("\n=====runtime所有字段和值=======")
        print(f"runtime类型:{type(runtime).__name__}")
        print(f"runtime完整内容:{runtime}")
        print(f"=========={runtime.context}==============")
        print(f"========={runtime.context.user_id}============")
        return None
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent=create_agent(
        model=model,
        tools=[get_user_greeting,get_user_settings],
        system_prompt="""你是一个个性化助手，可以根据用户设置提供定制化服务""",
        context_schema=AppContext,  #context_schema告诉create_agent，agent的所有工具都可以通过runtime.context访问这个类型的实例，在agent.invoke时，需要用context参数传入该类型的实例
        middleware=[print_state],
    )
    #调用
    result = agent.invoke(
        {"messages":[("user","我的设置时什么")]},
        config={"callbacks":[langfuse_handler]},
        context=AppContext(user_id="U001",user_name="张三",language="zh",timezone="Asia/Shanghai"),
    )
    print(f"AI:{result["messages"][-1].content}")


def example_2():
    """
    动态运行时上下文
    知识点：
    - 读取State
    - 更新State
    - State的生命周期
    """
    print("\n========示例2: 动态运行时上下文State============")
    from langchain.agents.middleware import AgentState,after_model
    from langgraph.runtime import Runtime
    from langgraph.types import Command
    from typing import Any,NotRequired

    class AppState(AgentState):
        """自定义应用状态"""
        conversation_topic: NotRequired[str]
        user_preferences: NotRequired[str]

    #工具访问State
    @tool
    def get_state_info(runtime:ToolRuntime) ->str:
        """获取当前回话的状态信息"""
        state = runtime.state
        topic = state.get("conversation_topic","未设置")
        return f"当前话题:{topic}"
    #工具更新state
    @tool
    def set_topic(new_topic:str,runtime:ToolRuntime)->Command:
        """
        设置对话话题
        参数：
            new_topic: 新的话题
        返回：
            状态更新命令
        """
        """
        当工具需要同时更新状态和响应LLM时，
        - conversation_topic 更新自定义状态字段
        - messages 添加工具响应消息到对话历史
        """
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "conversation_topic":new_topic,
                "messages":[
                    ToolMessage(
                        content=f"对话已设置为:{new_topic}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    #中间件
    @after_model(state_schema=AppState)
    def track_state(state:AppState,runtime:Runtime)->dict[str,Any]|None:
        """记录状态变化"""
        topic = state.get("conversation_topic",None)
        if topic:
            print(f"[状态追踪】当前话题：{topic}")
        return None

    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent=create_agent(
        model=model,
        tools=[get_state_info,set_topic],
        system_prompt="你是一个对话管理助手，可以设置和查询当前对话话题",
        middleware=[track_state],
    )
    print("\n=======测试1:查询当前状态===========")
    result1 = agent.invoke(
        {"messages":[("user","当前话题时什么")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n=====测试2:设置话题=====")
    result2 = agent.invoke(
        {"messages":["user","把话题设置为'Python编程'"]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")


def example_3():
    """
    示例3: 跨对话上下文（Store）
    目标：持久化存储
    知识点：
    - Store读写
    - 跨回话数据共享
    - 用户偏好存储
    """
    print("\n=============示例3:跨对话上下文（Store)==========")
    from langgraph.store.memory import InMemoryStore
    #创建内存存储
    store = InMemoryStore()
    """
    InMemoryStore时langgraph提供的内存存储实现
    警告：
    数据仅存在于内存，进程退出后全部丢失
    适合开发和测试，不适合生产
    
    生成环境使用
    - AsyncSqliteStore SQLite文件持久化
    - PostgresStore （PostgreSQL数据库）
    
    
    Checkpointer vs store 架构对比
    - checkpointer： InMemorySaver，单次会话内（多轮对话），Agent完整状态图，消息，节点状态；通过thread_id进行标识，传入方式 create_agent(checkpointer="")
    - store: InMemoryStore,跨会话，长期记忆，任意业务数据，用户偏好，历史；通过namespace+key进行标识，记住用户长期偏好，下次登录仍记得，传入方式create_agent(store="")
    """
    #预存一些用户偏好
    store.put(("preferences",),"user_001",{"theme":"dark","language":"zh","notifications":True})
    store.put(("preferences",),"user_002",{"theme":"light","language":"en","notifications":False})
    """
    store.put(namespace,key,value)
    namespace:元组，类似文件夹路径
    key: 字符串
    value：任意JSON可序列号数据
    
    preferences: 根文件夹
    key="user_001" 是文件名
    value: 实际的字典数据
    
    不同的key，就会新增记录
    """

    @dataclass
    class UserContext:
        user_id:str

    @tool
    def get_user_preferences(runtime:ToolRuntime[UserContext])->str:
        """获取用户偏好设置"""
        user_id = runtime.context.user_id
        pref = store.get(("preferences",),user_id)
        print(pref)
        """
        StoreItem的属性：
        .value  --- 存储的原始数据
        .key。  key字符串
        .namespace  namespace元组
        .create_at 创建时间戳
        """
        if pref:
            return f"用户{user_id}偏好设置:\n"+"\n".join(f"-{k}:{v}" for k,v in pref.value.items())
        return f"用户{user_id}暂无偏好设置"

    @tool
    def save_user_preference(key:str,value:str,runtime:ToolRuntime[UserContext])->str:
        """
        保存用户偏好设置
        参数：
            key：偏好键名
            value: 偏好值
        返回：
            保存结果
        """
        user_id= runtime.context.user_id
        pref = store.get(("preferences",),user_id)
        preferences = pref.value if pref else {}
        preferences[key] = value
        store.put(("preferences",),user_id,preferences)
        return f"已存储：{key} = {value}"

    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent=create_agent(
        model=model,
        tools=[get_user_preferences,save_user_preference],
        system_prompt="你是一个用户偏好管理助手",
        store=store,
    )
    print("\n=====测试1:查询偏好=======")
    result1 = agent.invoke(
        {"messages":[("user","我的偏好设置是什么？")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="user_001")
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n=====测试2:保存用户偏好======")
    result2 = agent.invoke(
        {"messages":[("user","帮我把字体大小设置为’大'")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="user_001")
    )
    print(f"AI:{result2['messages'][-1].content}")



def example_4():
    """
    上下文生命周期管理
    目标：理解不同上下文的生命周期
    知识点：
    - Context 单次调用生命周期
    - State 对话生命周期
    - Store 永久生命周期
    :return:
    """
    print("\n========示例4:上下文生命周期管理===========")
    @dataclass
    class SessionContext:
        "会话上下文，单词调用有效"
        session_id: str
        request_source:str
    #Store用于永久存储
    lifecycle_store = InMemoryStore()
    @tool
    def log_request(runtime: ToolRuntime[SessionContext])->str:
        """
        记录请求日志
        Context 在单次请求中有效
        """
        ctx = runtime.context
        message = f"[{ctx.session_id}] 来源:{ctx.request_source}"
        #存储到Store
        lifecycle_store.put(("logs",),ctx.session_id,{"source":ctx.request_source})
        return f"请求已记录：{message}"
    @tool
    def get_request_history(runtime: ToolRuntime[SessionContext])->str:
        """获取请求历史，从Store"""
        #store.search()
        """
        store.search(namespace_prefix) 按namespace的前缀搜索所有匹配的条目
        参数：
          namespace_prefix 元组，前缀匹配
        返回
            List[StoreItem]   --- StoreItem列表
            每个item同样有key，value
        与store.get()有区别
            get -- 需要精确到namespace+key
            search --- 只要namespace，不需要key
        """
        print(runtime)
        print("\n==========================")
        print(lifecycle_store)
        logs = lifecycle_store.search(("logs",))
        if not logs:
            return "暂无历史记录"
        return f"共{len(logs)}条记录"

    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent=create_agent(
        model=model,
        tools=[log_request,get_request_history],
        system_prompt="你是一个请求日志的管理助手",
        context_schema=SessionContext,
        store=lifecycle_store,
    )
    print("\n======测试1:记录请求=======")
    result1 = agent.invoke(
        {"messages":[("user","记录这次请求")]},
        config={"callbacks":[langfuse_handler]},
        context=SessionContext(session_id="user_001",request_source="web")
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n======测试2:查看历史=============")
    result2 = agent.invoke(
        {"messages":[("user","查看请求历史")]},
        config={"callbacks":[langfuse_handler]},
        context=SessionContext(session_id="user_002",request_source="mobile")
    )
    print(f"AI:{result2['messages'][-1].content}")
def example_5():
    """
    个性化推荐系统
    知识点：
    - 多层上下文整合
    - 用户图像构建
    - 个性化推荐
    - 完整生产级实现
    :return:
    """
    print("\n========示例5:个性化推荐系统===========")
    @dataclass
    class UserContext:
        """用户上下文"""
        user_id: str
        user_level: str
    user_store = InMemoryStore()
    products_db = [
        {"id": 1, "name": "Python入门编程", "category": "编程", "price": 99, "level": "new"},
        {"id": 2, "name": "大模型应用开发实战", "category": "AI", "price": 199, "level": "regular"},
        {"id": 3, "name": "数据分析高薪训练营", "category": "精品", "price": 299, "level": "vip"},
        {"id": 4, "name": "Java后端进阶教程", "category": "编程", "price": 129, "level": "regular"},
        {"id": 5, "name": "AI提示词工程精通课", "category": "AI", "price": 159, "level": "vip"},
        {"id": 6, "name": "前端全栈精品课", "category": "精品", "price": 369, "level": "new"}
    ]
    @tool
    def view_history(runtime:ToolRuntime[UserContext])->str:
        """浏览历史"""
        user_id = runtime.context.user_id
        history = user_store.get(("history",),user_id)
        if history:
            return f"浏览历史：{','.join(history.value)}"
        return "暂无浏览记录"
    @tool
    def record_view(product_name:str,runtime:ToolRuntime[UserContext])->str:
        """
        记录浏览记录
        参数：
            product_name: 商品名称
        :param product_name:
        :param runtime:
        :return:
        """
        user_id = runtime.context.user_id
        history = user_store.get(("history",),user_id)
        views = history.value if history else []
        views.append(product_name)
        user_store.put(("history",),user_id,views)
        return f"已记录浏览：{product_name}"
    @tool
    def get_recommendotions(runtime:ToolRuntime[UserContext])->str:
        """根据用户等级和历史获取推荐"""
        ctx = runtime.context
        user_id = ctx.user_id
        user_level = ctx.user_level
        history =  user_store.get(("history",),user_id)
        viewed = history.value if history else []
        #根据等级过滤
        level_products =[p for p in products_db if p["level"] == user_level]
        #排除已浏览的
        recommended = [ p for p in level_products if p["name"] not in viewed]
        if not recommended:
            return "暂无推荐商品"
        output = f"为您推荐:({user_level}专属：\n"
        for p in recommended[:4]:
            output += f"- {p['name']}:¥{p['price']} ({p['category']})\n"
        return output
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[view_history,record_view,get_recommendotions],
        system_prompt="""
        你是一个个性化商品推荐助手
        - 根据用户等级推荐商品
        - 记录用户浏览历史
        - 推荐未浏览的商品
        """,
        context_schema=UserContext,
        store=user_store,
    )
    print("\n========测试1:新用户体验=========")
    result1 = agent.invoke(
        {"messages":[("user","有什么推荐的？")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="user_001",user_level="regular")
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n========测试2:浏览商品============")
    result2 = agent.invoke(
        {"messages":[("user","我想看看数据分析高级课程")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="user_002",user_level="regular")
    )
    print(f"AI:{result2['messages'][-1].content}")

    print("\n=======测试3:问浏览历史==========")
    result3 = agent.invoke(
        {"messages":[("user","我看过哪些")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="user_001",user_level="regular")
    )
    print(f"AI:{result3['messages'][-1].content}")

    print("\n========测试4:再次推荐，排除已浏览=============")
    result4 = agent.invoke(
        {"messages":[("user","还有什么其他推荐？")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="user_001",user_level="regular")
    )
    print(f"AI:{result4['messages'][-1].content}")

def example_6():
    """
    Store持久化，基于SqliteStore
    - SqliteStore初始化
    - 数据持久化道磁盘
    :return:
    """
    from langgraph.store.sqlite import SqliteStore
    import sqlite3
    db_path = "./lesson_08_store.db"
    conn = sqlite3.connect(db_path,check_same_thread=False,isolation_level=None)
    store = SqliteStore(conn=conn)
    @dataclass
    class UserProfile:
        """用户配置上下文"""
        user_id:str
        username:str
    @tool
    def save_profile(name:str,age:int,runtime:ToolRuntime[UserProfile]) ->str:
        """
        保存用户资料
        参数：
            name: 用户姓名
            age: 用户年龄
        """
        user_id = runtime.context.user_id
        profile = {"name":name,"age":age,"saved_at":"2026-07-02"}
        store.put(("profiles",),user_id,profile)
        return f"已保存{name}的资料，持久化到磁盘"

    @tool
    def load_profile(runtime:ToolRuntime[UserProfile]) ->str:
        """获取用户资料"""
        user_id = runtime.context.user_id
        profile = store.get(("profiles",),user_id)
        if profile:
            data = profile.value
            return f"用户:{data['name']},年龄：{data['age']}，保存时间:{data['saved_at']}"
        return "暂无资料，请先保存"
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[save_profile,load_profile],
        system_prompt="你是一个用户资料管理助手",
        context_schema=UserProfile,
        store=store,
    )
    print("\n==========测试1:保存资料写入磁盘==========")
    result1 = agent.invoke(
        {"messages":[("user","保存我的资料：姓名李那，年龄28")]},
        config={"callbacks":[langfuse_handler]},
        context=UserProfile(user_id="U001",username="李那"),
    )
    print(f"AI:{result1['messages'][-1].content}")
    #
    print("\n========测试2:读取磁盘，从磁盘读取=============")
    result2 = agent.invoke(
        {"messages":[("user","我的资料是什么?")]},
        config={"callbacks":[langfuse_handler]},
        context=UserProfile(user_id="U001",username="李那"),
    )
    print(f"AI:{result2['messages'][-1].content}")
def example_7():
    """
    Store持久化PostgreSql
    :return:
    """
    print("\n==============示例7:Store持久化：PostgreSQL==========")
    """
    pip install psycopg psycopg-pool
    pip install langchain-postgres psycopg2-binary
    pip install langgraph-checkpoint-postgres
    docker run -d --name langgraph-pg -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin123 -e POSTGRES_DB=ai_memory -p 5435:5432 postgres
    """
    from langgraph.store.postgres import PostgresStore
    DB_URI = "postgresql://admin:admin123@172.16.181.128:5435/ai_memory"
    try:
        import psycopg
        conn = psycopg.connect(
            conninfo=DB_URI,
            autocommit=True,
            sslmode="disable",
            connect_timeout=10
        )
        store = PostgresStore(conn=conn)
        store.setup()  #setup初始化表结构，prefix，key,value

        @dataclass
        class UserProfile:
            user_id:str
            username:str
        @tool
        def save_profile_pg(name:str,age:int,runtime:ToolRuntime[UserProfile]) ->str:
            """
            保存用户资料道postgresql
            参数：
                name: 用户名
                age: 用户年龄
            """
            user_id = runtime.context.user_id
            store.put(("profiles",),user_id,{"name":name,"age":age})
            return f"已保存{name}到Postgresql数据库"
        @tool
        def load_profile_pg(runtime:ToolRuntime[UserProfile]) ->str:
            """从Postgresql读取用户资料"""
            user_id = runtime.context.user_id
            profile = store.get(("profiles",),user_id)
            if profile:
                return f"用户：{profile.value['name']},年龄:{profile.value['age']}"
            return '暂无记录'
        model = init_chat_model(
            base_url=os.getenv('ARK_BASE_URL'),
            api_key=os.getenv('ARK_API_KEY'),
            model_provider="openai",
            model="Doubao-Seed-2.0-lite"
        )
        agent = create_agent(
            model=model,
            tools=[save_profile_pg,load_profile_pg],
            system_prompt="你是一个使用postgresql，作为记忆存储的助手",
            context_schema=UserProfile,
            store=store,
        )
        print("\n========测试1:写入并读取======")
        result1 = agent.invoke(
            {"messages":[("user","保存我叫王五，今年26岁，然后读取我的名字")]},
            config={"callbacks":[langfuse_handler]},
            context=UserProfile(user_id="U0002",username="王五")
        )
        print(f"AI:{result1['messages'][-1].content}")
    except Exception as e:
        print(f"连接数据库失败：{e}")


def main(example_number:int):
    print("="*60)
    print("第4课:提示词和中间件")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4,
        5:example_5,
        6:example_6,
        7:example_7
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(7)



