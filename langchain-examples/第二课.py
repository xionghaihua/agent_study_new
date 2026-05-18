from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from dataclasses import dataclass
from langchain_core.tools import tool
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
import os

from openai import base_url
from sympy.physics.units import temperature

load_dotenv()

langfuse_handler = CallbackHandler()

def example_1():
    """
    temperature:控制输出随机性
    max_tokens：限制输出长度
    timeout：超时设置
    :return:
    """
    print("\n============案例1:模型参数配置详解==========")
    model_deter= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model_deter,
        system_prompt="你是一个数学助手，只返回计算结果，不添加其他内容"
    )
    result1 = agent.invoke({"messages":[("user","22*27")]},config={"callbacks":[langfuse_handler]})
    print(f"回复:{result1['messages'][-1].content}")

    model_creative = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.8
    )
    agent2 = create_agent(
        model=model_creative,
        system_prompt="你是一个诗人，用诗意的语言回答任何问题"
    )
    result2 = agent2.invoke({"messages":[("user","描述一下春天的景色")]},config={"callbacks":[langfuse_handler]})
    print(f"回复:{result2['messages'][-1].content}")
def example_2():
    """
    消息类型：
    SystemMessage 系统消息
    HumanMessage 用户消息
    AIMessage AI消息
    :return:
    """
    print("\n============案例2:消息类型与多伦对话==========")
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        system_prompt="你是一个专业的翻译助手，擅长中英互译"
    )
    result1 = agent.invoke(
        {"messages":[HumanMessage("请翻译:hello,world")]},config={"callbacks":[langfuse_handler]}
    )
    print(f"用户:Hello,World")
    print(f"AI:{result1['messages'][-1].content}")
    #第二轮
    result2 = agent.invoke(
        {"messages":[
            SystemMessage("您是一个专业的中英翻译助手"),
            HumanMessage("请翻译:Hello,world"),
            AIMessage(result1['messages'][-1].content),
            HumanMessage("再翻译：Good，morning")
        ]},
        config={"callbacks":[langfuse_handler]}
    )
    #print(f"AI:{result2['messages'][-1].content}")
    for i,msg in enumerate(result2['messages']):
        print(f"消息{i+1}:[{msg.type}] {msg.content[:80]}...")

def example_3():
    """
    https://docs.langchain.com/oss/python/langchain/structured-output
    结构化输出
    response_format
    :return:
    """
    print("\n============案例3:结构化输出==========")
    @dataclass
    class WeatherResponse:
        """天气查询的结构化响应格式"""
        city:str
        temperature:float
        condition:str
        suggestion:str

    @tool
    def get_weather(city:str)->str:
        """
        查询指定城市的天气情况
        参数：
            city:城市名称
         返回：
            天气描述y:
        """
        weather_db={
            "上海":{"temp":22,"condition":"多云"},
            "北京":{"temp":25,"condition":"晴朗"},
            "广州":{"temp":28,"condition":"降雨"}
        }
        data = weather_db.get(city,{"temp":20,"condition":"未知"})
        return f"{city} {data['condition']},{data['temp']}"

    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你是一个天气助手",
        response_format=ToolStrategy(WeatherResponse)
    )
    result = agent.invoke(
        {"messages":[HumanMessage("北京天气怎么样")]},
        config={"callbacks":[langfuse_handler]}
    )

    print(f"AI回复:{result['messages'][-1].content}")
    if "structured_response" in result:
        structured = result["structured_response"]
        print(f"城市:{structured.city}")
        print(f"温度:{structured.temperature}度")
        print(f"天气:{structured.condition}")
def example_4():
    """
    流式输出
    stream()
    :return:
    """
    print("\n============案例4:流式输出==========")
    model= init_chat_model(
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY')
    )
    agent = create_agent(
        model=model,
        system_prompt="你是一个有创意的作家，擅长写短篇故事"
    )
    for chunk in agent.stream(
       {"messages":[HumanMessage("写一个关于AI的短篇故事，200字以内")]},
       stream_mode="messages",
       version="v2",
       config={"callbacks":[langfuse_handler]},
    ):
        #print(chunk)
        msg = chunk["data"][0]
        if msg.content:
            print(msg.content,end="",flush=True)
def example_5():
    """
    智能分析助手
    :return:
    """
    @tool
    def calc(expression:str)->str:
        """
        计算数学表达式
        参数:
         expression:数学表达式，如2*3=6
        返回:
          计算结果
        """
        try:
            result = eval(expression)
            return f"计算结果:{result}"
        except Exception as e:
            return f"计算有误:{e}"
    @tool
    def query_sales_data(region:str):
        """
        查询指定地区的销售数据
        参数：
            region: 地区名称
        返回:
            销售数据摘要
        """
        sales_db = {
            "华东": {"revenue": 150000, "orders": 3200, "growth": 15.5},
            "华南": {"revenue": 130000, "orders": 2800, "growth": 12.3},
            "华北": {"revenue": 120000, "orders": 2500, "growth": 9.8},
            "西南": {"revenue": 90000, "orders": 1800, "growth": 18.2}
        }
        data = sales_db.get(region)
        if data:
            return (
                f"{region}销售数据:\n"
                f"-营收:{data["revenue"]}\n",
                f"-订单数:{data["orders"]}\n",
                f"-增长率:{data["growth"]}%"
            )
        return f"未找到{region}的销售数据"
    model= init_chat_model(
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        temperature=0.3,
        max_tokens=800,
        timeout=30
    )
    agent = create_agent(
        model=model,
        tools=[calc, query_sales_data],
        system_prompt="你是一个专业的数据分析助手，擅长查询销售数据，擅长数学计算,请根据用户的问题，选择合适的工具进行分析，并输出专业建议"
    )
    result = agent.invoke(
        {"messages":[HumanMessage("帮我查询一下华东地区的销售情况")]},
        config={"callbacks":[langfuse_handler]}
    )
    print(f"AI回复:{result['messages'][-1].content}")



if __name__ == "__main__":
    #example_1()
    #example_2()
    #example_3()
    #example_4()
    example_5()