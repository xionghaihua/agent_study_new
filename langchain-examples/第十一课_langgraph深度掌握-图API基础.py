"""
目标： 掌握LangGraph图API的核心概念
知识点：
- 节点Nodes与边Edges
- 状态图StateGraph
- 消息图MessageGraph
- Thinking in LangGraph
- 状态定义，Annotated + Reducer
- 节点部分更新语义
- MessagesState 内置结构
"""
from http.client import responses

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from typing import TypedDict,Annotated,List
import operator
from dotenv import load_dotenv
import os


load_dotenv()
langfuse_handler = CallbackHandler()

def example_1():
    """
    示例1: 理解langgraph的基本结构
    知识点：
    - stategraph创建
    - 添加节点
    - 添加边
    - 编译和运行
    """
    print("\n======示例1:最简图-hello world======")
    #定义状态
    class SimpleState(TypedDict):
        messages: Annotated[List[str],operator.add]
        """
        Annotated[type,reducer]是python的类型提示扩展
        第一个参数：类型，list[str]
        第二个参数：reducer参数，告诉langgraph如何合并更新
        
        langgraph图的核心是状态，每个节点可以读取和修改状态
        但多个节点可能同时修改状态，如何合并这些修改？
        operator.add 累加
        
        reducer作用：
        - 当节点返回，{"messages":["hello"]}时
        - langgraph不会用["hello"]替换整个messages列表
        - 而是调用operator.add(old_messages,["hello"])来追加
        - 结果 ["hello"]被追加到现有列表
        """
        #定义节点函数
    def node_greeting(state:SimpleState)->dict:
        """问候节点"""
        print("[节点] 执行问候节点")
        return {"messages":["您好，欢迎使用Langgraph"]}
    def node_farewell(state:SimpleState)->dict:
        print("【节点】执行告别节点")
        return {"messages":["再见，期待下次再见"]}
    #构建图
    graph_builder = StateGraph(SimpleState)
    #添加节点
    graph_builder.add_node("greeting", node_greeting)
    graph_builder.add_node("farewell", node_farewell)
    #添加边
    graph_builder.add_edge(START, "greeting")
    graph_builder.add_edge("greeting", "farewell")
    graph_builder.add_edge("farewell",END)
    #编译图
    graph = graph_builder.compile()
    #运行图
    print("\n-=====执行图======")
    result  = graph.invoke({"messages":[]},config={"callbacks":[langfuse_handler]})
    print(f"\n最终状态：{result}")

def example_2():
    """
    示例2: 条件边
    目标：掌握条件路由
    知识点：
    - 条件边
    - 动态路由
    - 分支逻辑
    :return:
    """
    from datetime import datetime
    print("\n====示例2:条件边--分支逻辑=======")
    class RouteState(TypedDict):
        question: str
        answer: str
    #路由函数
    def route_question(state:RouteState) ->str:
        """根据问题类型路由"""
        question = state["question"].lower()
        if "天气" in question:
            print("路由-->天气节点")
            return "weather_node"
        elif "时间" in question:
            print("路由-->时间节点")
            return "time_node"
        else:
            print("路由-->默认节点")
            return "default_node"
    #节点
    def weather_node(state:RouteState)->dict:
        return {"answer":"今天天气晴朗，25度"}
    def time_node(state:RouteState)->dict:
        return  {"answer": f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
    def default_node(state:RouteState)->dict:
        return {"answer": "抱歉，我只能回复天气和时间的相关问题"}
    #构建图
    builder = StateGraph(RouteState)
    builder.add_node("weather_node", weather_node)
    builder.add_node("time_node", time_node)
    builder.add_node("default_node", default_node)
    #条件边
    builder.add_conditional_edges(
        START,
        route_question,
        {"weather_node":"weather_node","time_node":"time_node","default_node":"default_node"},
    )
    """
    add_conditional_edges路由字典
    从那个节点出发
    route_function 路由函数
    路由字典 {"key":"target_node"}
    """
    builder.add_edge("weather_node",END)
    builder.add_edge("time_node",END)
    builder.add_edge("default_node",END)
    #编译
    graph = builder.compile()
    #测试不同的问题
    print("\n====测试1:天气问题=====")
    result1 = graph.invoke({"question":"今天天气怎么样","answer":""},config={"callbacks":[langfuse_handler]})
    print(f"回答:{result1["answer"]}")

    result2 = graph.invoke({"question":"当前时间几点了","answer":""},config={"callbacks":[langfuse_handler]})
    print(f"回答：:{result2["answer"]}")

    result3 = graph.invoke({"question":"什么是量子计算","answer":""},config={"callbacks":[langfuse_handler]})
    print(f"回答：:{result3["answer"]}")

def example_3():
    """
    示例3: 消息图--对话流程
    目标： 使用MessageState构造对话图
    知识点：
    - MessagesState
    - 多轮对话
    - 消息历史
    :return:
    """
    print("\n=====示例3:消息图—对话流程=======")
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    #定义节点
    def call_model(state:MessagesState)->dict:
        """调用LLM"""
        print(f"节点调用LLM....{state}")
        responses = model.invoke(state["messages"])
        return {"messages":[responses]}
        """
        MessagesState是langgraph提供的内置状态类
        class messagesState[TypedDict]:
            messages: Annotated[list[BaseMessage],add_messages]
        - messages: 消息列表，使用add_messages作为reducer
          - add_messages langgraph专用的消息合并函数
            - 自动去重，通过消息ID
            - 追加新消息
            - 处置ToolMessage与AIMessage的关联
        相比operator.add,add_messsages更智能
        """
    #构建图
    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_model)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    graph = builder.compile()
    #运行
    result = graph.invoke(
        {"messages":[HumanMessage("您好，简单介绍一个langgraph")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result["messages"][-1].content}")


def example_4():
    """
    目标：构建多步骤处理流水线
    知识点：
    - 节点串联
    - 数据流转
    - 状态累计
    :return:
    """
    print("\n=============示例4:多节点流水线============")
    class PipelineState(TypedDict):
        input_text:str
        processed_text:str
        summary:str
    #节点1:预处理
    def preprocess(state:PipelineState)->dict:
        text = state["input_text"]
        processed = text.strip().lower()
        print(f"[节点1]预处理：'{processed}'")
        return {"processed_text":processed}
    #节点2:分析
    def analyze(state:PipelineState)->dict:
        text = state['processed_text']
        word_count = len(text.split())
        char_count = len(text)
        analysis = f"词数：{word_count},字符数：{char_count}"
        print(f"[节点2]分析完成：{analysis}")
        return {"summary":analysis}

    #节点3:生成报告
    def generate_report(state:PipelineState)->dict:
        report =  f"文件分析报告:\n"
        report += f"原文:{state['input_text']}\n"
        report +=f"处理后:{state['processed_text']}\n"
        report +=f"统计：:{state['summary']}\n"
        print(f"[节点3]报告生成完成")
        return {"summary":report}

    builder = StateGraph(state_schema=PipelineState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("analyze", analyze)
    builder.add_node("report", generate_report)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "analyze")
    builder.add_edge("analyze", "report")
    builder.add_edge("report", END)
    graph = builder.compile()
    print("\n====测试流水线=====")
    result  = graph.invoke(
        {"input_text":"hello world! this is a langgraph Tutorial","processed_text":"","summary":""},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"\n{result['summary']}")






def example_5():
    """
    示例5:客服路由系统
    目标：创建完整的客服路由图
    知识点：
    - 复杂图结构
    - 多条件路由
    - 节点组合
    - 生产级实现
    :return:
    """
    print("\n=======示例5:客服路由系统=========")
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    class CustomState(MessagesState):
        category:str

    #分类节点
    def classify_request(state:CustomState)->dict:
        """分类用户请求"""
        last_msg = state["messages"][-1].content.lower()
        if "订单" in last_msg or "物流" in last_msg:
            category = "order"
        elif "退款" in last_msg or "退货" in last_msg:
            category = "refund"
        else:
            category = "general"
        print(f"[分类节点]类别：{category}")
        return {"category":category}
    #订单处理节点
    def handle_order(state:CustomState)->dict:
        """处理订单查询"""
        response = AIMessage("您好，关于订单查询，请提供订单号，我将为您查询物流信息。")
        print(f"[订单节点]处理订单查询")
        return {"messages":[response]}
    #退款处理节点
    def handle_refund(state:CustomState)->dict:
        """处理退款请求"""
        response = AIMessage("您好，关于退款，请告知您的订单号和退款原因")
        print(f"[退款节点]处理退款请求")
        return {"messages":[response]}
    #通用处理
    def handle_general(state:CustomState)->dict:
        """通用处理"""
        print(f"[通用处理]调用LLM....")
        response = model.invoke(state["messages"])
        return {"messages":[response]}
    #路由函数
    def route_by_category(state:CustomState)->str:
        return state["category"]
    #构建图
    builder = StateGraph(CustomState)
    builder.add_node("classify", classify_request)
    builder.add_node("order", handle_order)
    builder.add_node("refund", handle_refund)
    builder.add_node("general", handle_general)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {"order":"order","refund":"refund","general":"general"}
    )
    builder.add_edge("order",END)
    builder.add_edge("refund",END)
    builder.add_edge("general",END)

    graph = builder.compile()

    #测试
    print("\n=========订单查询=========")
    result1 = graph.invoke(
        {"messages":[HumanMessage("我的订单到哪了？")]},
        config = {"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n=======退款请求============")
    result2 = graph.invoke(
        {"messages":[HumanMessage("我要申请退款")]},
        config = {"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")
    result3 = graph.invoke(
        {"messages":[HumanMessage("你们的产品有哪些优势？")]},
        config = {"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result3['messages'][-1].content}")
def main(example_number:int):
    print("="*60)
    print("第11课：langgraph深度掌握---图API基础")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4,
        5:example_5,
        #6:example_6
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(5)


"""
状态流转：
读取
处理
返回

"""