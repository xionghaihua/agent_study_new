from typing import Any
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain.agents.middleware import AgentState,Runtime
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
load_dotenv()

langfuse_handler = CallbackHandler()

def example_1():
    print("\n======案例1:Agent架构深度解析======")
    @tool
    def search(query:str)->str:
        """搜索信息"""
        return f"搜索结果：关于{query}的相关信息....."
    @tool
    def calculate(expression:str)->str:
        """计算表达式"""
        try:
            return str(eval(expression))
        except Exception:
            return "计算有误"
    #初始化模型
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[search, calculate],
        system_prompt="你是一个助手，可以搜索信息和进行计算"
    )
    print("\n======Agent执行流程（逐步显示）=======")
    for step in agent.stream(
 {"messages":[("user","搜索Python的最新版本，然后计算2+3")]},
        stream_mode="updates",
        version="v2",
        config={"callbacks":[langfuse_handler]},
    ):
        """
        agent.stream: 以流式方式逐步返回Agent的执行过程
        stream_model="updates",表示每次只返回最新变化的数据
        """
        for node,data in step["data"].items():
            if data.get("messages"):
                msg = data["messages"][-1]
                print(f"\n[节点：{node}]")
                print(f"类型:{msg.type}")
                print(f"内容:{msg.content[:150]}....")
    result = agent.invoke({"messages":[HumanMessage("计算10乘以5")]},config={"callbacks":[langfuse_handler]})
    print(f"AI:{result["messages"][-1].content}")


def example_2():
    print("\n======案例2:create agent参数详解======")
    @tool
    def get_time()->str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #中间件
    from langchain.agents.middleware import before_model
    from langgraph.runtime import Runtime
    from typing import Any
    @before_model
    def add_timestrap(state:AgentState,runtime:Runtime)->dict[str,Any] | None:
        """在每条消息前添加时间戳"""
        print(f"[中间件】处理消息，当前状态消息数{len(state["messages"])}")
        return None

    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[get_time],
        system_prompt="你是一个时间助手",
        middleware=[add_timestrap],
        name="time_assistant" #Agent 名称
    )
    result=agent.invoke({"messages":[HumanMessage("现在几点？")]},config={"callbacks":[langfuse_handler]})
    print(f"AI:{result["messages"][-1].content}")

def example_3():
    """
    Agent状态管理
    - 访问Agent内部状态
    - 自定义状态字段
    - 状态持久化基础

    AgentState是langgraph的内置状态类型，默认包含messages字段（消息历史），通过继承AgentState可以添加自定义字段
    class CustomAgentState(AgentState):
        tool_call_count: NotRequired[int]
    NotRequired:表示字段不是必须存在，可以不存在状态中
    """
    print("\n======案例3:Agent状态管理======")
    from langchain.agents.middleware import after_model
    from typing_extensions import NotRequired
    class CustomAgentState(AgentState):
        tool_call_count: NotRequired[int]
        user_preference: NotRequired[str]
    #中间件：统计工具调用次数
    """
    @after_model(state_schema=CustomAgentState)
    @after_model是中间件装饰器，在模型生产消息后执行
    模型生成消息--->@after_model中间件---->进入下一步
    state_schema参数：
    指定中间件使用的状态类型
    传入CustomAgentState后，state参数就会有tool_call_count和user_preference的类型提示
    """
    @after_model(state_schema=CustomAgentState)
    def trace_tool_usage(state:CustomAgentState,runtime:Runtime)->dict[str,Any] | None:
        """统计工具调用次数"""
        tool_calls = state.get("tool_call_count",0)
        for msg in state["messages"]:
            if hasattr(msg,"tool_calls") and msg.tool_calls:
                tool_calls += len(msg.tool_calls)
        return {"tool_call_count":tool_calls}
        return None
    #工具
    @tool
    def search_info(query:str) ->str:
        """搜索信息"""
        return f"关于{query}的信息...."
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[search_info],
        middleware=[trace_tool_usage],
        system_prompt="你是一个信息查询助手"
    )
    print("\n=====测试：状态追踪=======")
    result = agent.invoke({"messages":[("user","搜索python"),("user","再搜索一下java")]},config={"callbacks":[langfuse_handler]})
    print(f"工具调用次数:{result.get('tool_call_count',0)}")
    print(f"消息总数:{len(result['messages'])}")

def main(example_number:int):
    print("="*60)
    print("第五课:agent架构与创建")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(3)
