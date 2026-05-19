"""
核心组件：tools
掌握工具的高级定义技巧和运行时上下文访问
--工具定义的最佳实践
--自定义工具名称和描述
--高级参数
--运行时上下文访问
--工具错误处理
"""
import warnings
import logging

from langgraph.prebuilt import ToolRuntime

#过滤pydantic和langgraph的序列化告警
warnings.filterwarnings("ignore",category=UserWarning,module="pydantic")
warnings.filterwarnings("ignore",message=".*Pydantic serializer.*")
warnings.filterwarnings("ignore",message=".*Pre-structured output.*")
logging.getLogger("pydantic").setLevel(logging.ERROR)

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel,Field
from typing import Optional,Literal
from dataclasses import dataclass
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

langfuse_handler = CallbackHandler()

def example_1():
    print("\n========案例1:工具定义的最佳实践===========")
    #方式1: 基础工具定义
    @tool
    def search_news(keyword:str,limit:int=5)->str:
        """
        搜索最新新闻
        这个函数用于检索与关键词相关的新闻
        参数：
            keyword：搜索关键词，例如AI，科技
            limit: 返回结果的数量，默认为5
        返回：
            新闻摘要列表
        """
        news_db = {
            "AI": ["AI技术突破，GPT-5发布","AI在医疗领域的应用","AI助手成为日常标配"],
            "科技":["量子计算新发展","5G网络覆盖全球","区块链技术成熟应用"]
        }
        results = news_db.get(keyword,["暂无此新闻"])
        return "\n".join(results[:limit])
    #自定义工具名称
    #当函数名不够清晰，可以自定义工具名
    @tool("weather_query")
    def get_weather_info(city:str):
        """
        查询城市天气
        参数：
            city:城市名称
        返回：
            天气描述
        """
        weather_db = {
            "上海": {"temp": 22, "condition": "多云"},
            "北京": {"temp": 25, "condition": "晴朗"},
            "广州": {"temp": 28, "condition": "降雨"}
        }
        data = weather_db.get(city, {"temp": 20, "condition": "未知"})
        return f"{city} {data['condition']},{data['temp']}"
    #方式3:自定义名称和描述
    @tool("calc",description="执行数学运算，适用于所有数学计算问题")
    def calculate(expression:str)->str:
        """数学计算公式"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误:{e}"
    #初始化模型
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    #创建agent
    agent = create_agent(
        model=model,
        tools=[search_news,get_weather_info,calculate],
        system_prompt="你是一个多功能助手，可以查询新闻，天气和进行数学运算"
    )
    result1 = agent.invoke(
        {"messages":[HumanMessage("最近有什么AI相关的新闻")]},
        config={"callbacks":[langfuse_handler]}
    )

    print(f"AI: {result1["messages"][-1].content}")

    result2 = agent.invoke(
        {"message":[HumanMessage("北京天气怎么样")]},
        config={"callbacks":[langfuse_handler]}
    )
    #print(result2)
    print(f"AI: {result2["messages"][-1].content}")

    result3 = agent.invoke(
        {"messages":[HumanMessage("计算25*20的值为多少")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI: {result3["messages"][-1].content}")


def example_2():
    """
    高级参数Schema
    使用Pydantic定义复杂的工具参数
    :return:
    """
    print("\n========案例2:通过args_schema自定义参数===========")
    #定义输入Schema
    class TravelQuery(BaseModel):
        """旅行查询的输入参数"""
        origin:str = Field(description="出发城市")
        destination:str = Field(description="目的城市")
        date:str = Field(description="出发日期，格式：YYYY-MM-DD")
        travel_class:Literal["经济舱","商务舱","头等舱"] = Field(default="经济舱",description="舱位等级")
    #使用args_schema定义复杂的参数
    @tool(args_schema=TravelQuery)
    def search_flights(origin:str,destination:str,date:str,travel_class:str="经济舱"):
        """
        查询航班信息，这个工具用于搜索两个城市之间的航班
        :param origin:
        :param destination:
        :param date:
        :param travel_class:
        :return:
        """
        flights_db = {
            ("北京", "上海"): [
                {"time": "08:00", "price": 1200, "airline": "国航"},
                {"time": "10:30", "price": 1050, "airline": "东航"},
                {"time": "14:15", "price": 980, "airline": "南航"},
                {"time": "19:00", "price": 820, "airline": "吉祥航空"}
            ],
            ("北京", "广州"): [
                {"time": "07:30", "price": 1350, "airline": "国航"},
                {"time": "11:20", "price": 1180, "airline": "南航"},
                {"time": "15:40", "price": 1020, "airline": "深航"},
                {"time": "20:10", "price": 900, "airline": "春秋航空"}
            ],
            ("上海", "深圳"): [
                {"time": "09:00", "price": 1100, "airline": "东航"},
                {"time": "12:50", "price": 960, "airline": "海航"},
                {"time": "16:30", "price": 880, "airline": "吉祥航空"},
                {"time": "21:00", "price": 750, "airline": "中联航"}
            ],
            ("成都", "杭州"): [
                {"time": "08:20", "price": 950, "airline": "川航"},
                {"time": "13:10", "price": 830, "airline": "国航"},
                {"time": "17:45", "price": 720, "airline": "厦航"}
            ],
            ("重庆", "南京"): [
                {"time": "10:00", "price": 860, "airline": "西部航空"},
                {"time": "14:25", "price": 740, "airline": "南航"},
                {"time": "18:50", "price": 650, "airline": "东航"}
            ]
        }
        route = (origin,destination)
        flights = flights_db.get(route,[])
        if not flights:
            return f"暂无{origin}到{destination}的航班"
        price_multiplier = {"经济舱":1.0,"商务舱":2.5,"头等舱":4.0}
        multiplier = price_multiplier.get(travel_class,1.0)
        result = []
        for f in flights:
            price = int(f["price"]*multiplier)
            result.append(f"{f['airline']} {f['time']}-{price} {travel_class}")
        return f"{origin}--->{destination}{date}:\n" + "\n".join(result)
    #初始化模型
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    #创建agent
    agent = create_agent(
        model=model,
        tools=[search_flights],
        system_prompt="你是一个专业的旅游规划助手，可以帮助用户查询航班信息，请根据用户需求，选择合适的出行建议"
    )
    result1 = agent.invoke(
        {"messages":[HumanMessage("帮我查一下北京到上海，明天早上出发，商务舱")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI: {result1["messages"][-1].content}")
def example_3():
    """
    运行时上下文访问（ToolRuntime)
    访问对话历史
    访问用户上下文
    流式写入
    :return:
    """
    print("\n========案例3:运行时上下文访问===========")
    import warnings
    warnings.filterwarnings("ignore",message="Pydantic serializer warnings")
    #定义用户上下文
    class UserContext(BaseModel):
        """用户上下文信息"""
        user_id:str
        user_name:str
    #访问对话历史
    @tool
    def get_conversation_history(runtime:ToolRuntime):
        """
        ToolRuntime：是浪chain在执行工具函数时候自动注入的运行上下文
        :param runtime:
        :return:
        runtime.context 用户自定义上下文
        runtime.state。AgentState状态
        runtime.stream_writer
        """
        messages = runtime.state["messages"]
        """
        runtime.state是AgentState类型，本质是一个字典，包含Agent的完整运行时状态
        state["messages"]存储了完整的对话历史，每个消息都具备msg.type，msg.content
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        user_messages = []
        for msg in reversed(messages):
            if hasattr(msg,"type") and msg.type == "human":
                user_messages.append(msg.content)
                if len(user_messages)>=3:
                    break
        if user_messages:
            return "最新用户消息:\n"+"\n".join(f"-{msg}" for msg in reversed(user_messages))
        return "暂无历史消息"
    #访问用户上下文
    @tool
    def get_user_profile(runtime:ToolRuntime[UserContext]):
        """
        表示runtime.context类型为UserContext
        :param runtime:
        :return:
        context=UserContext() 传入
        """
        user_id= runtime.context.user_id
        user_name = runtime.context.user_name
        user_db = {
            "001": {"name": "张娜", "level": "VIP", "points": 5000},
            "002": {"name": "李明", "level": "普通会员", "points": 1200},
            "003": {"name": "王芳", "level": "VIP", "points": 4800},
            "004": {"name": "赵强", "level": "钻石会员", "points": 8600},
            "005": {"name": "刘梅", "level": "普通会员", "points": 950},
            "006": {"name": "陈宇", "level": "VIP", "points": 5200}
        }
        user = user_db.get(user_id,[])
        if user:
            return (
                f"用户信息:\n"
                f"-姓名:{user.get("name",user_name)}\n"
                f"-等级:{user.get("level","未知")}\n"
                f"-积分:{user.get("points",0)}\n"
            )
        return  f"未知用户:{user_id}的信息"
    # 初始化模型
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    # 创建agent
    agent = create_agent(
        model=model,
        tools=[get_conversation_history,get_user_profile],
        system_prompt="你是一个多智能助手，可以查看对话历史，获取用户信息等",
        context_schema=UserContext,
    )
    result1 = agent.invoke(
    {"messages":[HumanMessage("我的个人信息是什么")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_id="002",user_name="李明")
    )
    print(f"AI: {result1['messages'][-1].content}")

if __name__ == "__main__":
    #example_1()
    #example_2()
    example_3()