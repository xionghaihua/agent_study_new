"""
记忆系统-短期记忆
掌握短期记忆，会话内记忆的实现
知识点：
- 对话历史管理
- checkpointer基础
- 会话状态维护
- 记忆生命周期
- thread_id详解
- checkpointer 参数详解
"""

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

from sympy import checkodesol, resultant

load_dotenv()
langfuse_handler=CallbackHandler()

def example_1():
    """
    示例1:多轮对话
    目标：理解短期记忆的基本用法
    知识点：
    - InMemorySave的使用
    - thread_id的使用
    - 对话连续性
    """
    print("\n==========示例1:-多轮对话============")
    #创建记忆管理器
    checkpointer = InMemorySaver()
    """
    InMemorySaver数据存储在内存中，进程退出后丢失，生产环境建议用PostgresSaver
    """
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent = create_agent(
        model=model,
        system_prompt="你是一个聊天助手，记住我们聊过的内容",
        checkpointer=checkpointer,
    )
    #config={"configurable":{"thread_id":"chat_001"}}
    print("\n======第一轮对话=========")
    result1 = agent.invoke(
        {"messages":[("user","我叫张三，今年28岁")]},
        config={"configurable":{"thread_id":"chat_001"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n======第二轮对话==============")
    result2 = agent.invoke(
        {"messages":[("user","你还记得我叫什么吗？")]},
        config={"configurable":{"thread_id":"chat_001"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")

def example_2():
    """
    示例2:多会话隔离
    目标：掌握不同会话的记忆隔离
    知识点：
    - thread_id 隔离
    - 兵法会话管理
    - 会话切换
    """
    print("\n=======示例2:对会话隔离===============")
    checkpointer = InMemorySaver()
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent = create_agent(
        model=model,
        system_prompt="你是一个友好的助手",
        checkpointer=checkpointer,
    )

    result_a1 = agent.invoke(
        {"messages":[("user","我是张三，我喜欢编程")]},
        config={"configurable":{"thread_id":"user_zhangsan"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result_a1['messages'][-1].content}")


    result_a2 = agent.invoke(
        {"messages":[("user","我是李四，我喜欢音乐")]},
        config={"configurable":{"thread_id":"user_lisi"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result_a2['messages'][-1].content}")
    result_a2 = agent.invoke(
        {"messages":[("user","我喜欢什么")]},
        config={"configurable":{"thread_id":"user_zhangsan"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result_a2['messages'][-1].content}")
    result_b2 = agent.invoke(
        {"messages":[("user","我喜欢什么")]},
        config={"configurable":{"thread_id":"user_lisi"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result_b2['messages'][-1].content}")


def example_3():
    """
    示例3:对话摘要与记忆优化
    目标：优化长对话的记忆管理
    知识点：
    - 对话长度管理
    - 消息压缩
    - 关键信息提取
    :return:
    """
    print("\n=======示例3:对话摘要与记忆优化=============")
    from langchain.agents.middleware import before_model,AgentState
    from langgraph.runtime import Runtime
    from typing import Any
    #中间件，限制消息数量防止上下文过长
    @before_model(can_jump_to=["end"])
    def limit_conversation_length(state:AgentState,runtime:Runtime) -> dict[str,Any]|None:
        """当消息过多时，提醒用户开始新会话"""
        from langchain.messages import AIMessage
        if len(state["messages"]) > 6:
            return {"messages":[AIMessage("对话历史较长，建议开始新的对话以保持上下文清晰")],"jump_to":"end"}
        return None

    checkpointer=(
        InMemorySaver()
    )
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent = create_agent(
        model=model,
        system_prompt="你是一个聊天助手，如果对话很长，主动建议用户开启新对话",
        checkpointer=checkpointer,
        middleware= [limit_conversation_length],
    )
    print("\n----多轮对话测试------")
    for i in range(4):
        result=agent.invoke(
            {"messages":[("user",f"这是第{i+1}次对话")]},
            config={"configurable":{"thread_id":"long_chat_001"},"callbacks":[langfuse_handler]},
        )
        print(f"轮次{i+1}:{len(result['messages'])}条消息，AI回复内容:{result['messages'][-1].content}")

def example_4():
    """
    示例4: 任务状态追踪
    目标：使用自定义State+Command追踪任务进度
    知识点：
    - 自定义AgentState扩展
    - 工具通过Cammand更新State
    - 中间件追踪状态变化
    - 多轮对话中的状态的持续
    :return:
    """
    from langchain.agents.middleware import AgentState,after_model
    from langchain.tools import ToolRuntime
    from langgraph.runtime import Runtime
    from langgraph.types import Command
    from langchain_core.messages import ToolMessage
    from typing import Any,NotRequired

    class TaskState(AgentState):
        """任务状态：追踪当前任务名和进度"""
        current_step:NotRequired[int]
        task_name: NotRequired[str]
    #每次模型响应后，打印丹铅任务状态
    @after_model(state_schema=TaskState)
    def track_task_state(state:TaskState,runtime:Runtime)->dict[str,Any]|None:
        task_name = state.get("task_name")
        current_step = state.get("current_step")
        if task_name:
            print(f"[状态]任务:{task_name} | current_step:{current_step}")
        return  None
    #工具，开始任务（写入State）
    @tool
    def start_task(task_name:str,runtime:ToolRuntime)->Command:
        """
        开始一个新任务
        参数：
            task_name: 任务名称
        """
        tool_call_id = runtime.tool_call_id
        return Command(
            update={
                "task_name":task_name,
                "current_step":1,
                "messages":[
                    ToolMessage(
                        content=f"任务{task_name}已创建，当前进度:第1步",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    @tool
    def complete_step(step_number:int,runtime:ToolRuntime)->Command:
        """
        完成任务的某个步骤
        参数：
          step_number:步骤编号
        """
        state = runtime.state
        current_task = state.get("task_name","未知任务")
        tool_call_id = runtime.tool_call_id
        next_step = step_number + 1
        return Command(
            update={
                "current_step":next_step,
                "messages":[
                    ToolMessage(
                        content=f"步骤{step_number}已完成，{current_task} 当前进度:第{next_step}步",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    @tool
    def get_progress(runtime:ToolRuntime)->str:
        """获取当前进度"""
        state = runtime.state
        task_name = state.get("task_name")
        current_step = state.get("current_step")
        if task_name:
            return f"当前任务:{task_name},已完成{current_step}步"
        return "暂无运行中的任务"

    checkpointer=InMemorySaver()
    model= init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent=create_agent(
        model=model,
        tools=[start_task,complete_step,get_progress],
        system_prompt="""
        你是一个任务管理助手，帮助用户创建任务并追踪进度，使用工具来更新任务进度
        """,
        checkpointer=checkpointer,
        middleware=[track_task_state],
    )
    config={"configurable":{"thread_id":"task_001"}}
    print("\n======第一轮：创建任务========")
    result1 = agent.invoke(
        {"messages":[("user","帮我创建一个学习python的任务")]},
        config={**config,"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n=====第二轮:完成第1步=====")
    result2 = agent.invoke(
        {"messages":[("user","完成第1步")]},
        config={**config,"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")
    print("\n======第三轮:查询进度========")
    result3 = agent.invoke(
        {"messages":[("user","我现在的进度如何")]},
        config={**config,"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result3['messages'][-1].content}")
    print("\n=======第四轮:完成第2步=======")
    result4 = agent.invoke(
        {"messages":[("user","完成第2步")]},
        config={**config,"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result4['messages'][-1].content}")


def example_5():
    print("\n======示例5:预约管理系统=========")
    checkpointer = InMemorySaver()
    appointments = {}
    @tool
    def create_appointment(date:str,time:str,purpose:str)->str:
        """
        创建预约
        参数：
            date: 日期
            time: 时间
            purpose:预约目的
        返回：
            预约结束
        """
        appointment_id = len(appointments) + 1
        appointments[appointment_id] = {"date":date,"time":time,"purpose":purpose}
        return f"预约已创建\nID:{appointment_id}\n时间:{date}{time}\n目的:{purpose}"
    @tool
    def list_appointments()->str:
        """列出所有预约"""
        if not appointments:
            return "暂无预约记录"
        output = "预约列表:\n"
        for aid,apt in appointments.items():
            output += f"{aid},{apt["date"]} {apt["time"]} {apt["purpose"]}\n"
        return output
    @tool
    def cancel_appointment(appointment_id:str)->str:
        """
        取消预约
        参数：
            appointment_id：预约ID
        """
        if appointment_id in appointments:
            del appointments[appointment_id]
            return f"预约{appointment_id}已取消"
        return f"未找到预约{appointment_id}"
    model=init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent=create_agent(
        model=model,
        tools=[create_appointment,list_appointments,cancel_appointment],
        system_prompt="""你是一个预约管理助手，可以
        -创建新的预约
        -查看所有预约
        -取消预约
        """,
        checkpointer=checkpointer,
    )
    print("\n======测试1:创建预约=====")
    result1 = agent.invoke(
        {"messages":[("user","帮我预约明天下午3点去人民医院看医生")]},
        config={"configurable":{"thread_id":"task_001"},"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n====测试2:查看预约====")
    result2 = agent.invoke(
        {"messages":[("user","我有哪些预约？")]},
        config={"configurable": {"thread_id": "task_001"}, "callbacks": [langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")

    print("\n======测试3:取消预约========")
    result3 = agent.invoke(
        {"messages":[("user","帮我取消ID为1的预约")]},
        config={"configurable": {"thread_id": "task_001"}, "callbacks": [langfuse_handler]},
    )
    print(f"AI:{result3['messages'][-1].content}")

def example_6():
    print("\n======示例6:持久化短期记忆=============")
    """
    - postgresSaver
    - thread_id隔离与持久化
    - 生产环境配置
    按thread_id隔离不同会话，存的是对话过程，而不是用户画像，agent不会跨thread_id
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    from psycopg.rows import dict_row
    DB_URI = "postgresql://admin:admin123@172.16.181.128:5435/ai_memory"
    try:
        #创建连接池，生产环境推荐
        #如果本地单线程，也可以直接用psycopg.connect()
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold":0,
            "row_factory":dict_row,
        }
        pool = ConnectionPool(
            conninfo=DB_URI,
            max_size=5, #连接池最大连接数，限制兵法请求数量
            kwargs=connection_kwargs
        )
        #初始化postgresSaver
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  #首次运行必需调用，创建checkpoint表结构
        """
        setup()做了啥？
        在postgresql创建checkpoint相关表
        - checkpoints 存储每个thread检查点
        - checkpoint_writes 存储待写入的更新
        - checkpoint_blob 存储大对象
        """
        model = init_chat_model(
            base_url=os.getenv('ARK_BASE_URL'),
            api_key=os.getenv('ARK_API_KEY'),
            model_provider="openai",
            model="Doubao-Seed-2.0-lite",
            temperature=0
        )
        agent = create_agent(
            model=model,
            system_prompt="你是一个聊天助手，记住我们聊过的内容",
            checkpointer=checkpointer,
        )
        print("\n=====测试1:自我介绍========")
        result1 = agent.invoke(
            {"messages":[("user","我叫张三，今年28岁，喜欢跳舞")]},
            config={"configurable":{"thread_id":"pg_chat_001"},"callbacks":[langfuse_handler]},
        )
        print(f"AI:{result1['messages'][-1].content}")

        print("\n======测试2:测试记忆读取数据========")
        result2 = agent.invoke(
            {"messages":[("user","你还记得我叫什么？喜欢什么吗？")]},
            config={"configurable":{"thread_id":"pg_chat_001"},"callbacks":[langfuse_handler]},
        )
        print(f"AI:{result2['messages'][-1].content}")
    except Exception as e:
        print(f"连接数据库失败：{e}")





def main(example_number:int):
    print("="*60)
    print("第8课-记忆系统之短期记忆")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4,
        5:example_5,
        6:example_6
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(6)