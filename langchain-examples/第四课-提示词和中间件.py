"""
提示词工程
--静态提示词模版
- 动态提示词 @dynamic_prompt
- AgentState和runtime
- 中间件的7种hook
- wrap_model_call wrap_tool_call
"""

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from langchain.agents.middleware import (
    before_agent,
    before_model,
    after_model,
    after_agent,
    wrap_model_call,
    wrap_tool_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
    AgentState,
)
from langgraph.runtime import Runtime
from typing import Any, Callable
from dotenv import load_dotenv
import os

from sympy.physics.units import temperature

load_dotenv()
langfuse_handler = CallbackHandler()

def example_1():
    print("\n======案例1:静态提示词设计最佳实践=====")
    SYSTEM_PROMPT = """你是一个专业的编程助手，专注于回答编程相关的问题
你的职责:
- 提供准确、实用的代码示例
- 解释代码的工作原理
- 指出最佳实践和潜在问题
回答要求
- 代码必须包含详细注释
- 优先使用Python 3.10+语法
如果用户的问题超出编程范围，请礼貌地引导回编程话题"""


    @tool
    def search_code_example(language:str,topic:str) ->str:
        """
        搜索代码示例
        参数：
            language:编程语言（如python，javascript）
            topic: 主题关键词
        返回：
            相关代码示例
        """
        examples = {
            ("python","排序"): "sorted([3,1,2]) #返回 [1，2，3]",
            ("python","文件"): "with open('file.txt') as f: content = f.read()"
        }
        return examples.get((language,topic),f"暂无{language}的{topic}示例")

    #初始化模型
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    #创建agent
    agent = create_agent(
        model=model,
        tools=[search_code_example],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke(
        {"messages":[("user","Python种如何对列表进行排序")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result["messages"][-1].content}")



def example_2():
    print("\n======案例1:动态提示词-根据上下文生成提示=====")
    """
    使用@dynamic_prompt装饰器创建动态提示词
    - @dynamic_prompt
    - 根据运行时上下文生成提示
    """
    from dataclasses import dataclass
    @dataclass
    class UserContext:
        """用户上下文"""
        user_level:str

    """
    dynamic_prompt装饰器，讲普通函数转换为动态提示词中间件，每次模型调用前执行，返回一个字符串作为提示词
    @dynamic_prompt
    def my_prompt(request:ModelRequest) ->str:
       return "生成的提示词"
    request对象
    - request.state : AgentState 对话状态
    - request.runtime: Runtime运行环境
    """
    @dynamic_prompt
    def generate_adaptive_prompt(request:ModelRequest) ->str:
        """根据用户水平生成适应性提示词"""
        user_level="beginner"
        if request.runtime and request.runtime.context:
            user_level=request.runtime.context.user_level
        base_prompt="你是一个python编程高手"
        if user_level == "beginner":
            return """{base_prompt}
用户水平:初学者
- 使用简单易懂的语言，避免专业术语
- 提供详细的逐步解释和类比"""
        elif user_level == "intermediate":
            return """{base_prompt}
用户水平:中级
- 可以用专业术语
- 关注最佳实践和设计模式
"""
        else:
            return """{base_prompt}
用户水平:专家
- 深入讨论底层原理和架构设计
"""
    @tool
    def explain_concept(concept:str) ->str:
        """解释编程概念"""
        explanations = {
            "装饰器": "装饰器是Python种修改函数行为的高级功能",
            "闭包": "闭包是函数和其引用环境的组合"
        }
        return explanations.get(concept,f"暂无{concept}的解释")
    #初始化模型
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[explain_concept],
        middleware=[generate_adaptive_prompt],
    )
    result1 = agent.invoke(
        {"messages":[("user","什么是装饰器")]},
        config={"callbacks":[langfuse_handler]},
        context=UserContext(user_level="beginner"),
    )
    print(f"AI:{result1['messages'][-1].content}")

def example_3():
    print("\n======案例3:AgentState和Runtime详解=====")
    """
    - AgentState是什么
    - Runtime是什么
    - 如何在中间件中读写他们
    
    AgentState是LangChain中间件的内存，它是TypedDict字典，在Agent整个生命周期中存在，每次模型调用前后，中间件都能读写这个状态
    内置字段
    messages: list[AnyMessage] 对话历史列表 新消息追加而非替换
    jump_to: 可选 由中间件设置，控制执行流跳转
    - tools 跳转到工具节点
    - model 跳转到模型节点
    - end 终止Agent
    structured_response(可选）： 由response_format参数决定，存放解析后的结构化对象
     
     扩展自定义字段
     class DemoState(AgentState):
        call_count: NotRequired[int]
    
     访问方式
     state["messages"]
     state.get("call_count",0)
     
     ------------
     Runtime是什么
     --------------
     Runtime是Agent执行时的运行时环境对象
     每次agent.invoke或agent.stream调用时创建
     中间件和动态提示词通过它访问执行环境
     核心属性：
     runtime.context  用户自定义上下文，通过invoke的context=参数传入
     runtime.model 当前使用的模型实例
    """
    from typing_extensions import NotRequired
    from dataclasses import dataclass

    #用户上下文类
    @dataclass
    class DemoContext:
        user_id:str
        verbose:bool = True

    #扩展AgentState，添加自定义字段
    class DemoState(AgentState):
        call_count: NotRequired[int]  #记录模型调用次数 NotRequired字段可选
    #before_model，after_model 执行多次
    @before_model(state_schema=DemoState)
    def count_calls(state:DemoState,runtime:Runtime)->dict[str,Any] | None:
        """统计模型调用次数"""
        msg_count = len(state["messages"])
        current = state.get("call_count",0)
        user_id = runtime.context.user_id if runtime.context else "unknown"
        if runtime.context and runtime.context.verbose:
            print(f"[count_calls] 用户{user_id},消息数：{msg_count},调用次数:{current}")
        return {"call_count":current+1}
    @after_model(state_schema=DemoState)
    def show_result(state:DemoState,runtime:Runtime)->dict[str,Any] | None:
        """显示模型返回结果"""
        last_msg = state["messages"][-1]
        content = last_msg.content if hasattr(last_msg,"content") else None
        total = state.get("call_count",0)
        print(f"[show_result] 回复长度：{len(content)}字符，累计调用：{total}次")
        return None

    #before_agent Agent执行前执行，整个生命周期只一次
    #after_agent Agent执行后执行，整个生命周期只一次
    @before_agent(state_schema=DemoState)
    def init_state(state:DemoState,runtime:Runtime)->dict[str,Any] | None:
        """Agent启动时初始化"""
        print("[before_agent]Agent开始执行")
        return None
    @after_agent(state_schema=DemoState)
    def cleanup_state(state:DemoState,runtime:Runtime)->dict[str,Any] | None:
        total = state.get("call_count",0)
        print(f"[after_agent] Agent执行完成，总调用次数{total}")
        return None

    @tool
    def demo_tool(query:str) ->str:
        """演示工具"""
        return f"演示结束：{query}"
    #初始化模型
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite"
    )
    agent = create_agent(
        model=model,
        tools=[demo_tool],
        system_prompt="你是一个演示助手",
        middleware=[init_state,count_calls,show_result,cleanup_state],
        context_schema=DemoContext
    )
    result = agent.invoke(
        {"messages":[("user","您好")]},
        config = {"callbacks":[langfuse_handler]},
        context = DemoContext(user_id="user_001",verbose=True),
    )
    print(f"\n最终call_count:{result.get('call_count',0)}")
    print(f"AI:{result["messages"][-1].content}")
    """
    Langchain 中间件完整生命周期
    
    @before_agent  整个生命周期只执行一次，Agent执行前
    ReAct循环(可能多次):
        wrap_model_call 包装整个模型调用
        before_model 每次模型调用前
           调用模型
        after_model 每次模型调用后
        wrap_tool_call 工具执行
    @after_agent 整个生命周期只执行一次
    @dynamic_prompt 动态提示词
    """


def example_4():
    print("\n======案例4:wrap_model_call模型调用包装=====")
    """
    掌握wrap_model_call中间件
    - 拦截模型调用
    - 实现重试逻辑
    - 动态模型选择
    完全控制模型调用过程的中间件
    接收ModelRequest和handler可调用对象
    必须调用handler(request),否则模型不会执行
    
    handler的作用
    - 单个wrap_model_call时，handler指向实际模型调用
    - 多个wrap_model_call，handler指向下一个中间件，形成链式调用
    - request.override(model=...)创建新请求副本，替换指定字段
    
    """
    @wrap_model_call
    def retry_on_failure(
            request: ModelRequest,
            handler: Callable[[ModelRequest],ModelResponse],
    )->ModelResponse|None:
        """模型调用失败自动尝试"""
        max_retries = 3
        for attempt in range(1,max_retries+1):
            try:
                print(f"[重试中间件] 第{attempt}次尝试")
                return handler(request)
            except Exception as e:
                if attempt == max_retries:
                    print(f"[重试中间件]达到最大充实次数：{e}")
                print(f"[重试中间件]失败，准备重试：{e}")

    simple_model=init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    advanced_model=init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-pro",
        temperature=0.7,
    )
    @wrap_model_call
    def dynamic_model_selector(
            request:ModelRequest,
            handler: Callable[[ModelRequest],ModelResponse],
    )->ModelResponse:
        """根据对话长度选择模型"""
        if len(request.state["messages"])>10:
            print("[动态模型]使用高级模型(长对话)")
            selected_model=advanced_model
        else:
            print("[动态模型]使用简单模型(短对话)")
            selected_model=simple_model
        return handler(request.override(model=selected_model))
    @tool
    def calculate_complexity(problem:str)->str:
        """评估问题复杂度"""
        return f"问题{problem}的复杂度:中等"
    agent=create_agent(
        model=simple_model,
        tools=[calculate_complexity],
        system_prompt="你是一个问题分析助手",
        middleware=[retry_on_failure,dynamic_model_selector],
    )
    print("\n---测试：动态模型选择")
    result = agent.invoke(
        {"messages":[("user","帮我分析一下，如何设计一个分布式系统？")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result['messages'][-1].content}")

def main(example_number:int):
    print("="*60)
    print("第4课:提示词和中间件")
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
