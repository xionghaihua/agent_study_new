"""
langgraph实施了流式系统来显示实时更新，从而实现响应迅速且透明的用户体验
langgraph的流式传输系统可将图形运行的实时反馈显示到您的应用中
流式输出在langgraph中的重要性
    用户立即看到反馈
    减少等待时间
    节省内存使用
    提升用户体验
stream()同步
astream()异步

stream_mode
values: 在图的每个步骤之后流式传输状态的完整值，看到完整状态
updates: 将图的每个步骤之后的更新流式传输到状态。如果在同一步骤中进行了多个更新，则这些更新将分别流式传输，看到变化部分 （常用）
messages: 从调用LLM的任何图形节点流式传输2元组，看到AI逐字输出 （常用）
checkpoints: 返回检查点的完整状态内容
tasks: 返回任务开始结束，错误，结果的内容

v2版本： 统一不同流式输出的返回格式，每个数据块都有一个StreamPart结构一致的字典
"""
from typing import Annotated,TypedDict,List,Optional
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.types import Send
import operator
##====values,updates,debug模式=============
class State(TypedDict):
    numbers: List[int]
    results: Annotated[list[int],operator.add]
    final_sum: int

#分发数字
def split_numbers(state:State):
    numbers = state["numbers"]
    #每个数字分发给一个worker
    return [ Send("worker",{"number":num}) for num in numbers ]

#work阶段，计算平方
def calculate_square(state: State):
    number = state["number"]
    square = number * number
    return {"results": [square]}

#reduce阶段，求和
def sum_results(state: State):
    results = state.get("results",[])
    total = sum(results)
    return {"final_sum":total}
#构建图
def create_simple_graph():
    graph =  StateGraph(state_schema=State)
    graph.add_node("spliter",lambda s: s) #分发器
    graph.add_node("worker",calculate_square) #工作节点
    graph.add_node("sum",sum_results)  #求和
    #连接节点
    graph.add_edge(START,"spliter")
    graph.add_conditional_edges("spliter",split_numbers,["worker"])
    graph.add_edge("worker","sum")
    graph.add_edge("sum",END)
    return graph.compile()

print("\n===============messages模式==================")
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()

model = init_chat_model(
    base_url=os.getenv("ARK_BASE_URL"),
    api_key=os.getenv("ARK_API_KEY"),
    model="openai:DeepSeek-V4-Flash",
    temperature=0
)
class MyState(TypedDict):
    question:str
    results:str
def generate_answer(state:MyState):
    question = state["question"]
    answer = model.invoke([
        {"role":"user","content":f"{question}"}
    ])
    return {"answer":answer.content}

def generate_answer1(state:MyState):
    answer = model.invoke([{"role":"user","content":f"您好"}])
    return {"answer":answer.content}

def create_llm_graph():
    graph = StateGraph(state_schema=MyState)
    graph.add_node("generate_answer",generate_answer)
    graph.add_node("generate_answer1",generate_answer1)
    graph.add_edge(START,"generate_answer")
    graph.add_edge("generate_answer","generate_answer1")
    graph.add_edge("generate_answer",END)
    return graph.compile()

print("\n===========自定义模式==============")
from langgraph.config import get_stream_writer
import time
class FileState(TypedDict):
    filename:str #文件名称
    content:str #文件内容
    word_count:int  #内容数量
    processed: bool #是否处理

def read_file(state:FileState):
    """读取文件"""
    writer = get_stream_writer()
    #发送开始信息
    writer({"step":"读取文件","status":"开始","progress":0})
    time.sleep(1)

    #发送进度信息
    writer({"step":"读取文件","status":"正在读取....","progress":50})
    time.sleep(1)

    #模拟文件内容
    content = "这是一个示例文件，包含一些文本内容"

    #发送完成信息
    writer({
        "step":"读取文件",
        "status":"完成",
        "progress":100,
        "data": {"size": len(content)}
    })
    return {"content":content}

def count_words(state:FileState):
    """统计字数"""
    writer= get_stream_writer()
    writer({"step":"统计数字","status":"开始","progress":0})
    time.sleep(0.5)
    writer({"step":"统计数字","status":"正在分析....","progress":30})
    time.sleep(0.5)
    writer({"step":"统计数字","status":"计算中.....","progress":70})
    time.sleep(0.5)

    #计算字数
    word_count = len(state["content"])
    writer({
        "step":"统计字数",
        "status":"完成",
        "progress":100,
        "data": {"word_count": word_count}
    })
    return {"word_count":word_count}

def finalize_processing(state:FileState):
    """完成处理"""
    writer = get_stream_writer()
    writer({"step":"完成处理","status":"生成报告","progress":50})
    time.sleep(1)

    writer({
        "step":"完成处理",
        "status":"全部完成",
        "progress":100,
        "data":{
            "filename": state["filename"],
            "total_chars": state["word_count"],
            "summary": f"文件{state['filename']}处理完成，共{state['word_count']}个字符"
        }
    })
    return {"processed": True}

#构建图
def create_custom_graph():
    graph = StateGraph(state_schema=FileState)
    graph.add_node("read_file",read_file)
    graph.add_node("count_words",count_words)
    graph.add_node("finalize_processing",finalize_processing)
    graph.add_edge(START,"read_file")
    graph.add_edge("read_file","count_words")
    graph.add_edge("count_words","finalize_processing")
    return graph.compile()

print("\n===========checkpoint,tasks模式==============")
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()
model2 = init_chat_model(
    base_url=os.getenv("ARK_BASE_URL"),
    api_key=os.getenv("ARK_API_KEY"),
    model="openai:DeepSeek-V4-Flash",
    temperature=0
)
#创建子图
def subplot(state:MessagesState) -> MessagesState:
    answer = state["messages"][-1].content
    summary_prompt = f'请用一句话总结下面这句话:\n\n答:{answer}'
    response = model2.invoke(summary_prompt)
    return {"messages":[response]}

summary_subgraph = (
    StateGraph(state_schema=MessagesState)
    .add_node("subplot",subplot)
    .add_edge(START,"subplot")
    .add_edge("subplot",END)
    .compile()
)
def llm_answer_node(state:MessagesState):
    answer = model2.invoke(state["messages"])
    return {"messages":[answer]}

checkpointer = InMemorySaver()
def create_check_tasks_graph():
    parent_graph = (
        StateGraph(state_schema=MessagesState)
        .add_node("llm_answer",llm_answer_node)
        .add_node("summary_subgraph",summary_subgraph)
        .add_edge(START,"llm_answer")
        .add_edge("llm_answer","summary_subgraph")
        .compile(checkpointer=checkpointer)
    )
    return parent_graph

def run_example():
    app = create_simple_graph()
    app1 = create_llm_graph()
    app2 = create_custom_graph()
    app3 = create_check_tasks_graph()

    #测试数据
    initial_state = {
        "numbers":[1,2,3,4,5],
        "results": [],
        "final_sum": 0
    }
    print("================values模式=============")
    for result  in app.stream(initial_state,stream_mode="values",version="v2"):
        print(result)
    """
            ===============values模式=============
        {'type': 'values', 'ns': (), 'data': {'numbers': [1, 2, 3, 4, 5], 'results': [], 'final_sum': 0}, 'interrupts': ()}
        {'type': 'values', 'ns': (), 'data': {'numbers': [1, 2, 3, 4, 5], 'results': [], 'final_sum': 0}, 'interrupts': ()}
        {'type': 'values', 'ns': (), 'data': {'numbers': [1, 2, 3, 4, 5], 'results': [1, 4, 9, 16, 25], 'final_sum': 0}, 'interrupts': ()}
        {'type': 'values', 'ns': (), 'data': {'numbers': [1, 2, 3, 4, 5], 'results': [1, 4, 9, 16, 25], 'final_sum': 55}, 'interrupts': ()}
    """
    print("===============updates模式=================") #常用
    for result in app.stream(initial_state,stream_mode="updates",version="v2"):
        print(result)
    """
    ===============updates模式=================
    {'type': 'updates', 'ns': (), 'data': {'spliter': {'numbers': [1, 2, 3, 4, 5], 'results': [], 'final_sum': 0}}}
    {'type': 'updates', 'ns': (), 'data': {'worker': {'results': [1]}}}
    {'type': 'updates', 'ns': (), 'data': {'worker': {'results': [4]}}}
    {'type': 'updates', 'ns': (), 'data': {'worker': {'results': [9]}}}
    {'type': 'updates', 'ns': (), 'data': {'worker': {'results': [16]}}}
    {'type': 'updates', 'ns': (), 'data': {'worker': {'results': [25]}}}
    {'type': 'updates', 'ns': (), 'data': {'sum': {'final_sum': 55}}}
    """

    print("==========================DEBUG模式============================") #输出调试信息
    for result in app.stream(initial_state,stream_mode="debug",version="v2"):
        print(result)

    print("==========================MESSAGE模式================") #常用的模式
    for chunk in app1.stream({"question":"什么是langgrah状态图?"},stream_mode="messages",version="v2"):
        if chunk["type"] == "messages":
            result,metadata = chunk["data"]
            if metadata["langgraph_node"] == "generate_answer":
                print(result.content,end="",flush=True)
            #print(metadata)

    print("====================CUSTOM自定义模式======================================")
    initial_state1 = {
        "filename": "example.txt",
        "content": "",
        "word_count": 0,
        "processed": False
    }
    for chunk in app2.stream(initial_state1,stream_mode="custom",version="v2"):
        if chunk["type"] == "custom":
            data = chunk["data"]
            step = data.get("step","")
            status = data.get("status","")
            progress = data.get("progress","")
            data_result = data.get("data",{})
            #显示进度
            progress_bar = "#"*(progress//10) +"="*(10-progress//10)
            print(f"\n[{step}] {status}")
            print(f"进度:[{progress_bar}] {progress}%")

            if data_result:
                for key,value in data_result.items():
                     print(f"{key}:{value}")
    print("===================checkp[oints,tasks模式===================")
    config = {"configurable":{"thread_id":"1"}}

    input_state = {
        "messages":[{"role":"user","content":"langgraph是什么？请用100个字介绍"}],
    }
    for chunk in app3.stream(
        input_state,
        config,
        stream_mode="tasks",
        version="v2",
        summary_subgraph=True #子图流式输出
    ):
        print(chunk)

if __name__ == "__main__":
    run_example()