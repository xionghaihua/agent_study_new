from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt, interrupt
from typing import TypedDict, Annotated, List, Dict
from dotenv import load_dotenv
import operator
import os

from torch.distributed.pipeline.sync.checkpoint import checkpoint

load_dotenv()
langfuse_handler = CallbackHandler()

def example_1():
    """
    示例1: 子图-图嵌套
    目标：理解子图的概念和使用
    知识点：
    - 创建子图
    - 子图与主图的集成
    - 状态隔离与传递
    :return:
    """
    print("\n=====示例1-图嵌套=======")
    #子图状态
    class SubGraphState(TypedDict):
        input_data: str
        sub_result: str
    #主图状态
    class MainState(TypedDict):
        input_data: str
        final_result: str
    #构建子图
    def sub_process(state:SubGraphState)->Dict:
        """子图处理节点"""
        print("[子图]处理中....")
        return {"sub_result":f"子图处理结束，处理了'{state['input_data']}'"}
    subgraph = StateGraph(SubGraphState)
    subgraph.add_node("process",sub_process)
    subgraph.add_edge(START,"process")
    subgraph.add_edge("process",END)
    #子图编译
    compiled_subgraph = subgraph.compile()

    #构建主图
    def main_node(state:MainState)->Dict:
        """主图节点，调用子图"""
        print("[主图]调用子图....")
        result = compiled_subgraph.invoke({"input_data":state["input_data"]},config={"callbacks":[langfuse_handler]})
        return {"final_result":result["sub_result"]}
    main_graph = StateGraph(MainState)
    main_graph.add_node("process_node",main_node)
    main_graph.add_edge(START,"process_node")
    main_graph.add_edge("process_node",END)
    graph = main_graph.compile()
    print("\n====测试：主图调用子图=========")
    result = graph.invoke({"input_data":"测试数据","final_result":""},config={"callbacks":[langfuse_handler]})
    print(f"最终结果：{result['final_result']}")

def example_2():
    """
    示例2:状态快照
    目标：掌握状态快照
    知识点：
    - 检查的系统
    - 状态历史
    """
    print("\n====示例2:时间旅行---状态快照==========")
    from langgraph.checkpoint.memory import InMemorySaver
    class TimeTravelState(TypedDict):
        step: int
        data: str
    def step_1(state:TimeTravelState)->Dict:
        print("[步骤1]执行.....")
        return {"step":1,"data":"步骤1的结果"}

    def step_2(state:TimeTravelState)->Dict:
        print("[步骤2]执行.....")
        return {"step":2,"data":state["data"] + "+ 步骤2的结果"}
    def step_3(state:TimeTravelState)->Dict:
        print("[步骤3]执行.....")
        return {"step":3,"data": state["data"] + "+ 步骤3的结果"}
    #构建图，带检查点
    builder = StateGraph(TimeTravelState)
    builder.add_node("step1",step_1)
    builder.add_node("step2",step_2)
    builder.add_node("step3",step_3)
    builder.add_edge(START,"step1")
    builder.add_edge("step1","step2")
    builder.add_edge("step2","step3")
    builder.add_edge("step3",END)

    #使用检查点保存数据
    checkpointer = InMemorySaver()  #短期记忆
    """
    Langgraph的检查点持久化机制
    有checkpointer
，   graph.invoke(...,config={"configurable":{"thread_id":"a1"}}
    生产用postgreSQL    
    """
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable":{"thread_id":"time_travel_001"}}
    print("\n===执行图========")
    result = graph.invoke({"step":0,"data":"初始数据"},config={**config,"callbacks":[langfuse_handler]})
    print(f"最终结果：{result['data']}")

    #查看历史
    states = list(graph.get_state_history(config))
    print(f"\n====状态历史----\n{states}")
    """
    get_state_history: 返回该thread_id下所有历史checkpoint的迭代器
    每个快照包含
    - values 该步骤完整状态字典
    - next 下一步骤执行的节点名
    - metadata 执行元信息
    """
    for i,state in enumerate(states):
        print(f"状态:{i}:step={state.values.get("step")},data={state.values.get("data",'')}")
    """
        ====示例2:时间旅行---状态快照==========    
        ===执行图========
        [步骤1]执行.....
        [步骤2]执行.....
        [步骤3]执行.....
        最终结果：步骤1的结果+ 步骤2的结果+ 步骤3的结果
        
        ====状态历史----
        [StateSnapshot(values={'step': 3, 'data': '步骤1的结果+ 步骤2的结果+ 步骤3的结果'}, next=(), config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab1b-6c8e-8003-0f415465b270'}}, metadata={'source': 'loop', 'step': 3, 'parents': {}}, created_at='2026-08-08T03:30:45.765840+00:00', parent_config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab18-6b60-8002-603b7833a9b3'}}, tasks=(), interrupts=()), StateSnapshot(values={'step': 2, 'data': '步骤1的结果+ 步骤2的结果'}, next=('step3',), config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab18-6b60-8002-603b7833a9b3'}}, metadata={'source': 'loop', 'step': 2, 'parents': {}}, created_at='2026-08-08T03:30:45.764581+00:00', parent_config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab15-6802-8001-85278f446084'}}, tasks=(PregelTask(id='ca28a894-cca9-f36b-3455-7e2c682b1037', name='step3', path=('__pregel_pull', 'step3'), error=None, interrupts=(), state=None, result={'step': 3, 'data': '步骤1的结果+ 步骤2的结果+ 步骤3的结果'}),), interrupts=()), StateSnapshot(values={'step': 1, 'data': '步骤1的结果'}, next=('step2',), config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab15-6802-8001-85278f446084'}}, metadata={'source': 'loop', 'step': 1, 'parents': {}}, created_at='2026-08-08T03:30:45.763266+00:00', parent_config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab11-682e-8000-03ddfd872c33'}}, tasks=(PregelTask(id='4a9605d5-41a3-2fa5-3973-3c1090e1b2e8', name='step2', path=('__pregel_pull', 'step2'), error=None, interrupts=(), state=None, result={'step': 2, 'data': '步骤1的结果+ 步骤2的结果'}),), interrupts=()), StateSnapshot(values={'step': 0, 'data': '初始数据'}, next=('step1',), config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab11-682e-8000-03ddfd872c33'}}, metadata={'source': 'loop', 'step': 0, 'parents': {}}, created_at='2026-08-08T03:30:45.761631+00:00', parent_config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab0f-6970-bfff-904e52c1e63e'}}, tasks=(PregelTask(id='586c0800-af0b-2853-29a7-29caa7ffe71f', name='step1', path=('__pregel_pull', 'step1'), error=None, interrupts=(), state=None, result={'step': 1, 'data': '步骤1的结果'}),), interrupts=()), StateSnapshot(values={}, next=('__start__',), config={'configurable': {'thread_id': 'time_travel_001', 'checkpoint_ns': '', 'checkpoint_id': '1f192d98-ab0f-6970-bfff-904e52c1e63e'}}, metadata={'source': 'input', 'step': -1, 'parents': {}}, created_at='2026-08-08T03:30:45.760846+00:00', parent_config=None, tasks=(PregelTask(id='b3aae90b-bbe7-4082-0b00-1a8b7cd75892', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'step': 0, 'data': '初始数据'}),), interrupts=())]
        状态:0:step=3,data=步骤1的结果+ 步骤2的结果+ 步骤3的结果
        状态:1:step=2,data=步骤1的结果+ 步骤2的结果
        状态:2:step=1,data=步骤1的结果
        状态:3:step=0,data=初始数据
        状态:4:step=None,data=
    """

def example_3():
    """
    示例3: 中断机制--人工审核
    目标：掌握interrupt()函数实现Human-in-the-Loop
    知识点:
    - interrupt()函数，动态中断
    - Command(resume=...) 恢复执行
    - 人工审核交互
    """
    print("\n====示例3:中断机制--人工审核==========")
    class ReviewState(TypedDict):
        action_details:str
        status:str
    def approval_node(state:ReviewState):
        """审核节点--使用interrupt()暂停等待人工决策"""
        #interrupt()会暂停执行，payload会返回给调用方
        decision = interrupt(
            {
                "question":"审批此操作？",
                "details": state["action_details"],
            }
        )
        print(f"拿到审核结果{decision}")
        #根据审核结果路由到不同的节点
        if decision:
            return Command(goto="approve")
        else:
            return Command(goto="reject") #goto跳到指定的节点
    def approve_node(state:ReviewState)->dict:
        print("审核通过")
        return {"status":"已批准"}
    def reject_node(state:ReviewState)->dict:
        print("审核拒绝")
        return {"status":"已拒绝"}
    #构建图
    builder = StateGraph(ReviewState)
    builder.add_node("approval", approval_node)
    builder.add_node("approve", approve_node)
    builder.add_node("reject", reject_node)

    builder.add_edge(START,"approval")
    builder.add_edge("approve",END)
    builder.add_edge("reject",END)
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    print("【情况1：审批通过】")
    config1 = {"configurable":{"thread_id":"review_pass_001"},"callbacks":[langfuse_handler]}
    result1 = graph.invoke(
        {"action_details":"申请办公用品¥500","status":"待审批"},
        config=config1
    )
    print(f"中断信息：{result1}")

    #模拟人工审核
    print("\n=====步骤2:人工审批==========")
    print(f"审批内容:{result1['action_details']}")
    confirm = input("是否批准？(y/n):").strip().lower()
    approved = confirm == "y" or confirm == "y"
    if approved:
        print("人工审批：通过")
    else:
        print("人工审批:拒绝")
    #使用Command(resume=...)恢复
    print("\n====步骤3:继续执行=======")
    result1 = graph.invoke(Command(resume=approved),config=config1)
    print(f"最终结果:{result1['status']}")


    config2 = {"configurable":{"thread_id":"review_pass_002"},"callbacks":[langfuse_handler]}
    result2 = graph.invoke(
        {"action_details":"申请购买笔记本电脑，费用¥12000","status":"待审批"},
        config=config2
    )

    #模拟人工审核
    print("\n=====步骤2:人工审批==========")
    print(f"审批内容:{result2['action_details']}")
    confirm2 = input("是否批准？(y/n):").strip().lower()
    approved2 = confirm2 == "y" or confirm2 == "y"
    if approved2:
        print("人工审批：通过")
    else:
        print("人工审批:拒绝")
    #使用Command(resume=...)恢复
    print("\n====步骤3:继续执行=======")
    result2 = graph.invoke(Command(resume=approved2),config=config2)
    print(f"最终结果:{result2['status']}")


def example_4():
    """
    示例4:综合实战--审批工作流
    目标：创建完整的审批工作流系统
    知识点：
    - 多阶段审批
    - 条件分支
    - interrupt实现人工审核
    - Command(resume=...)恢复执行
    :return:
    """
    print("\n=======示例4:综合实战--审批工作流==============")
    class ApprovalState(TypedDict):
        request:str
        amount:int
        approval_status:str
        final_decision:str
    def submit_request(state:ApprovalState)->dict:
        """提交请求"""
        print(f"[提交]收到请求:{state['request']},金额:${state['amount']:,}")
        return {"approval_status":"已提交"}
    def manager_review(state:ApprovalState)->Command:
        """经理审核"""
        amount = state["amount"]
        if amount <= 1000:
            print("[经理审核]金额较小，自动通过")
            return {"approval_status":"经理审核通过"}
        else:
            print("[经理审核]金额较大，需要总监审核")
            return Command(goto="director_review")



    def director_review(state:ApprovalState)->dict:
        """总监审核"""
        print("[总监审核]等待人工审核.....")
        decision = interrupt(
            {
                "question":"请审核此申请",
                "request":state["request"],
                "amount":state["amount"],
            }
        )
        if decision:
            print("[总监审核]审核通过")
            return {"approval_status": "总监审核通过"}
        else:
            print("[总监审核]审核拒绝")
            return {"approval_status": "总监审核拒绝"}
    def final_decision(state:ApprovalState):
        """最终决定"""
        status = state["approval_status"]
        if "拒绝" in status:
            decision = "审核拒绝"
        else:
            decision = "审核通过"
        print(f"[最终决定] {decision}")
        return {"final_decision": decision}
    #构建图
    builder = StateGraph(ApprovalState)
    builder.add_node("submit", submit_request)
    builder.add_node("manager_review", manager_review)
    builder.add_node("director_review", director_review)
    builder.add_node("final_decision", final_decision)

    builder.add_edge(START,"submit")
    builder.add_edge("submit","manager_review")
    builder.add_edge("manager_review","final_decision")
    builder.add_edge("director_review","final_decision")
    builder.add_edge("final_decision",END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    print("\n======测试1:小额申请(¥500)--经理自动通过========")
    config1 = {"configurable":{"thread_id":"review_pass_001"},"callbacks":[langfuse_handler]}
    result1 = graph.invoke(
        {"request":"购买办公用品","amount":800,"approval_status":"","final_decision":""},
        config=config1
    )
    print(f"结果:{result1['final_decision']}")

    print("\n======测试1:大额申请(¥50000)--需要总监审批========")
    config2 = {"configurable":{"thread_id":"review_pass_002"},"callbacks":[langfuse_handler]}
    result2 = graph.invoke(
        {"request":"购买汽车","amount":50000,"approval_status":"","final_decision":""},
        config=config2
    )
    #检查是否有中断
    if result2.get("__interrupt__"):
        interrupt_info = result2.get("__interrupt__")[0].value
        print(f"===中断信息====")
        print(f"申请内容: {interrupt_info['request']}")
        print(f"金额: {interrupt_info['amount']}")
        print(f"问题: {interrupt_info['question']}")
        #模拟人工审核
        print("\n=====步骤2:总监审核=====")
        confirm = input("是否批准？(y/n):").strip().lower()
        approved = confirm == "y" or confirm == "y"
        if approved:
            print("总监审核:通过")
        else:
            print("总监审核:拒绝")
        print("\n=====步骤3:继续执行=======")
        result2 = graph.invoke(
            Command(resume=approved),
            config=config2
        )
        print(f"结果:{result2['final_decision']}")
    else:
        print("未命中中断，直接完成")
        print(f"结果：:{result2['final_decision']}")
def main(example_number:int):
    print("="*60)
    print("第12课：langgraph子图等高级")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4,
        #5:example_5,
        #6:example_6
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(4)