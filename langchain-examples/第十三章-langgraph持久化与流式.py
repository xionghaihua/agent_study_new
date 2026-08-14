"""
目标：掌握langgraph的检查点持久化，流式输出和事件流
知识点：
- 检查点持久化（PostGresSaver)
- 流失输出（astream_events)
- 事件流处理
- 实时更新
- 中断与恢复
- 程序启动后恢复执行
- 前端集成模式
- create_agent vs StateGraph范式转换
"""
from typing import TypedDict, Annotated

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import  dict_row
from dotenv import load_dotenv
import asyncio
import os
load_dotenv()
langfuse_handler = CallbackHandler()

DB_URI = "postgresql://admin:admin123@172.16.181.128:5435/ai_memory"

"""工具函数：创建PostgresSaver检查点保存器"""
def get_checkpointer(db_uri:str=DB_URI,max_size:int=5,setup:bool=True)->PostgresSaver:
    """
    创建PostgresSaver检查点保存器（使用连接池)
    参数：
      db_uri：数据库连接字符串
      max_size：连接池最大连接数
      setup: 是否执行setup()创建表结构（首次运行需要）
    返回：
        PostgresSaver 实例
    使用示例：
     checkpointer = get_checkpointer()
     agent = create_agent(model,tools=[],checkpointer=checkpointer)
    """
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,#关闭预编译语句， 防止连接池在多进程/重启后出现报错
        "row_factory": dict_row,
    }
    pool = ConnectionPool(
        conninfo=db_uri,
        max_size=max_size,
        kwargs=connection_kwargs,
    )
    checkpointer = PostgresSaver(pool)
    if setup:
        checkpointer.setup()
    return checkpointer


def example_1():
    """
    检查点持久化--PostgresSaver
    目标：掌握使用PostgresSaver实现对话状态持久化
    知识点：
    - PostgresSaver在PostgresSQL中持久化检查点
    - 使用thread_id区分不同对话
    - 中断后恢复对话状态
    """
    print("\n=======示例1: 检查点持久化-PostgresSaver===========")
    from langchain.agents import create_agent
    from langchain.tools import tool
    #定义工具
    @tool
    def get_user_info(user_id:str)->str:
        """
        查询用户信息
        参数：
            user_id：用户ID
        返回：
            用户详细信息
        """
        user_db = {
            "u001": {"name":"张三","age":30,"city":"北京","level":"VIP"},
            "u002": {"name": "李四", "age": 35, "city": "上海", "level": "普通会员"},
            "u003": {"name": "王五", "age": 27, "city": "广州", "level": "黄金会员"},
        }
        user = user_db.get(user_id,{})
        if user:
            return f"姓名:{user['name']},年龄:{user['age']},城市:{user['city']}，等级:{user['level']}"
        return f"未找到用户{user_id}"
    @tool
    def get_order_history(user_id:str,limit:int=5) ->str:
        """
        查询用户的历史订单
        参数：
            user_id: 用户ID
            limit: 返回订单数量限制
        返回：
            订单历史记录
        """
        orders_db = {
            "u001":[
                {"order_id":"ORD1001","product":"iphone 15","amount": 8999,"date":"2024-01-10"},
                {"order_id": "ORD1002", "product": "AirPods Pro", "amount": 1999, "date": "2024-02-10"},
                {"order_id": "ORD1003", "product": "MacBook Air", "amount": 9999, "date": "2024-03-10"},
            ],
            "u002":[
                {"order_id":"ORD2001","product":"IPad Air","amount":4999,"date":"2024-01-20"},
            ],
        }
        orders = orders_db.get(user_id,[])
        if not orders:
            return f"用户{user_id}暂无订单记录"
        results = []
        for order in orders[:limit]:
            results.append(f"订单{order['order_id']}:{order['product']}-¥{order['amount']},{order['date']}")
        return "\n".join(results)
    #初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    checkpointer = get_checkpointer()
    agent = create_agent(
        model=model,
        tools=[get_user_info,get_order_history],
        system_prompt="你是一个电商客服助手，帮助用户查询个人信息与订单历史",
        checkpointer=checkpointer
    )
    """
    from langgraph.graph.message import  add_messages
    from langchain.messages import AnyMessage
    class AgentState(TypedDict):
        messages:Annotated[list[AnyMessage],add_messages]
    def agent_node(state:AgentState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    def tools_node(state:AgentState):
        tool_messages = state["messages"][-1].tool_calls
        return {"messages":tool_messages}
    def should_continue(state:AgentState):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END
    builder = StateGraph(AgentState)
    builder.add_node("agent",agent_node)
    builder.add_node("tools",tools_node)
    builder.add__edge(START,"agent")
    builder.add_conditional_edges("agent",should_continue,"tools")
    builder.add__edge("tools","agent")
    graph = builder.compile(checkpointer=checkpointer)
    """
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable":{"thread_id":thread_id},"callbacks":[langfuse_handler]}
    print("\n=========第一轮对话=============")
    result1 = agent.invoke(
        {"messages":[("user","帮我查询用户u001的信息")]},
        config=config
    )
    print(f"AI:{result1['messages'][-1].content}")

    #第二轮
    print("\n=======第二轮对话=============")
    result2 = agent.invoke(
        {"messages":[("user","他的订单有哪些?")]},
        config=config
    )
    print(f"AI:{result2['messages'][-1].content}")
    new_thread_id = str(uuid.uuid4())
    new_config = {"configurable":{"thread_id":new_thread_id},"callbacks":[langfuse_handler]}
    result3 = agent.invoke(
        {"messages":[("user","查询用户u002的订单?")]},
        config=new_config
    )
    print(f"AI:{result3['messages'][-1].content}")


def example_2():
    """
    示例2: 流式输出-astream
    掌握Langgraph的流式输出能力
    知识点：
    - stream()同步流式
    - stream_mode 参数详解
    - 实时获取生成内容
    """
    print("\n========示例2:流式输出-astream=======")
    from langchain.agents import create_agent
    from langchain.tools import tool
    @tool
    def write_essay(topic:str)->str:
        """
        根据主题写一遍文章摘要
        参数：
            topic: 文章主题
        返回：
            文章摘要内容
        """
        essay_db = {
            "人工智能": (
                "人工智能AI是计算机科学的一个分支，在创建能够执行通常需要人类智能的任务的系统\n"
                "近年来，深度学习技术的突破使AI在图像识别、自然语言处理和游戏等领域取得了显著成就。\n"
                "当前，大语言模型LLM正在改变人机交互的方式，成为生产力的重要工具。\n"
                "未来AI将在医疗，教育，交通等领域发挥更大的作用。"
            ),
            "气候变化":(
                "气候变化是当今世界面临的最大挑战之一。\n"
                "全球变暖导致极端天气频发，海平面上升和生态系统破坏。\n"
                "国际社会正在通过减少碳排放，发展可再生能源等方式应对这一挑战。\n"
                "每个人的行动都对减缓气候变化有着重要意义"
            ),
        }
        return essay_db.get(topic,f"关于'{topic}'的摘要：这是一个重要的话题，值的深入研究和讨论。")

    #初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    #创建agent
    agent = create_agent(
        model=model,
        tools=[write_essay],
        system_prompt="你是一个写作助手，帮助用户撰写文章摘要.",
    )
    #同步流式输出
    print("\n===========同步流式输出===========")
    print("用户: 帮我写一遍关于人工智能的摘要")
    print("\nAI:",end="",flush=True)
    first_chunk = True #标记是否是第一个chunk
    for chunk in agent.stream(
            {"messages":[("user","帮我写一遍关于人工智能的摘要")]},
        stream_mode="messages",
        config={"callbacks":[langfuse_handler]}
    ):
        """
        stream_mode="messages"
        messages: 按token流式输出
        values： 每次输出完整的状态快照（默认）
        updates: 每次输出状态的增量更新
        debug: 输出调试信息
        """
        if first_chunk:
            print(f"\n\nchunk:{chunk}")
            first_chunk = False
        msg = chunk[0]
        if msg.content:
            print(msg.content,end="",flush=True)
    print("\n\n======流式输出完成==========")
    print("\n======使用messages模式的流式输出===========")
    print("用户:写一篇气候变化的摘要")
    print("\nAI:", end="", flush=True)
    first_chunk = True
    for chunk in agent.stream(
            {"messages": [("user", "帮我写一篇气候变化的摘要")]},
            stream_mode="messages",
            version="v2",
            config={"callbacks": [langfuse_handler]}
    ):
        if first_chunk:
            print(f"\n\nchunk:{chunk}")
            first_chunk = False
        msg = chunk["data"][0]
        if msg.content:
            print(msg.content,end="",flush=True)
    print("\n\n=====messages模式流式输出完成==========")


def example_3():
    """
    示例3: 事件流---astream_events
    目标:掌握使用事件流监控Agent执行过程
    知识点:
    - astream_events方法
    - 不同类型的事件处理
    - 实时监控工具调用和模型输出
    """
    print("\n=======示例3: 事件流-astream_events=============")
    from langchain.agents import create_agent
    from langchain.tools import tool
    import asyncio
    @tool
    def analyze_data(metric:str,region:str)->str:
        """
        分析指定地区的业务指标数据
        参数：
            metric: 指标名称
            region: 地区名称
        """
        data_db = {
            # 营收 revenue
            ("revenue", "华东"): {"value": 150000, "growth": 15.5, "rank": 1},
            ("revenue", "华南"): {"value": 120000, "growth": 8.2, "rank": 2},
            ("revenue", "华北"): {"value": 98000, "growth": 5.3, "rank": 3},
            # users
            ("users", "华东"): {"value": 42000, "growth": 22.1, "rank": 1},
            ("users", "华南"): {"value": 31000, "growth": 10.5, "rank": 2},
            ("users", "华北"): {"value": 22000, "growth": 4.8, "rank": 3},
            ("users", "西南"): {"value": 18500, "growth": 14.3, "rank": 4},
            # orders
            ("orders", "华东"): {"value": 12600, "growth": 11.2, "rank": 1},
            ("orders", "华南"): {"value": 9800, "growth": 6.7, "rank": 2}
        }
        key = (metric,region)
        data = data_db.get(key)
        if data:
            return f"{region}地区 {metric}指标: 值={data['value']},增长率={data['growth']},排名=第{data['rank']}名"
        return f"暂无法查询{region}地区{metric}指标数据"
    #初始化模型
    model = init_chat_model(
        base_url="https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="openai:qwen3.7-plus",
        temperature=0
    )
    #创建agent
    agent = create_agent(
        model=model,
        tools=[analyze_data],
        system_prompt="你是一个数据分析助手，帮助用户分析业务指标",
    )
    #使用asyncio.run运行一步事件流
    async def run_event_stream():
        print("\n===事件流监控开始=====")
        print("用户：分析一下华东地区的revenue指标")
        print("\n[事件流输出]")
        print("="*40)
        first_event = True
        async for event in agent.astream_events(
                {"messages":[("user","帮我分析一下华东地区的revenue指标")]},
            version="v2",
            config={"callbacks":[langfuse_handler]}
        ):
            if first_event:
                print(f"\n\n完整内容：{event}")
                first_event = False
            event_type = event.get("event","")
            event_name = event.get("name","")
            event_data = event.get("data",{})
            if e
def main(example_number:int):
    print("="*60)
    print("第13课：langgraph持久化与流式")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        #4:example_4,
        #5:example_5,
        #6:example_6
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(2)