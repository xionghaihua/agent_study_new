#graph.update_state()更新图状态
"""
更新对应状态，就是手动修改线程里的状态数据，它不会删除历史，只会生成新的检查点。你可以改消息，改工具结果
加用户信息，删除内容，覆盖内容。更新完之后，下一次run就从新状态继续执行

#方法参数
config
必须包含thread_id执行更新的线程
可选包含checkpoint_id来分叉选定的检查点

values:用于更新状态的值
更新会传递给reducer函数，没有reducer的通道会被覆盖

as_node: 指定更新来自哪个节点，影响下一步执行的节点
"""
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict,Annotated
from operator import add

class TaskState(TypedDict):
    task_id:str
    title:str
    assignee:str #接收任务的人
    priority:int  #优先级
    comments: Annotated[list,add] #评论
    status: str  #状态
def create_task(state:TaskState):
    """创建任务"""
    return {
        "status": "进行中",
        "comments": [f"任务:'{state['title']}'已创建，分配给{state['assignee']}"]
    }
def update_task(state:TaskState):
    return {
        "status":"已更新",
        "comments": ["任务状态已更新"]
    }
def plan_task(state:TaskState):
    return {
        "status": "已准备",
        "comments": [f"任务 '{state['title']}' 已准备"]
    }

#创建任务管理流程
workflow = StateGraph(state_schema=TaskState)
workflow.add_node("create",create_task)
workflow.add_node("update",update_task)
workflow.add_node("complete",plan_task)
workflow.add_edge(START,"create")
workflow.add_edge("create","update")
workflow.add_edge("update","complete")
workflow.add_edge("complete",END)

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

#创建任务
import uuid
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id":thread_id}}
result = app.invoke({
    "task_id": "T001",
    "title": "开发注册功能",
    "assignee": "张三",
    "priority": 1,
    "comments": [],
    "status": "待分配"
},config=config)

print("======任务创建完成=======")
print(f"状态:{result['status']}")
print(f"评论: {result['comments']}")
"""
======任务创建完成=======
状态:已准备
评论: ["任务:'开发注册功能'已创建，分配给张三", '任务状态已更新', "任务 '开发注册功能' 已准备"]
"""

history = list(app.get_state_history(config))
#获取complete节点的上一个update节点
before_complete = next(s for s in history if s.next == ("complete",))
print("\n==使用as_node参数======")
#对update节点状态进行修改，实际上，是重新创建了一个检查点
app.update_state(before_complete.config,{
    "status":"已更新",
    "comments": ["自动通知任务优先级提升"],
    "priority": 3
})

#执行一次流程，如果修改完状态后，想让图从update节点执行起来，需要手动调用invoke方法
final_result = app.invoke(None, config)
print(final_result)
"""
======任务创建完成=======
状态:已准备
评论: ["任务:'开发注册功能'已创建，分配给张三", '任务状态已更新', "任务 '开发注册功能' 已准备"]

==使用as_node参数======
{'task_id': 'T001', 'title': '开发注册功能', 'assignee': '张三', 'priority': 3, 'comments': ["任务:'开发注册功能'已创建，分配给张三", '任务状态已更新', '自动通知任务优先级提升', "任务 '开发注册功能' 已准备"], 'status': '已准备'}
"""


#打印checkpoint
history = list(app.get_state_history(config=config))
for idx,snapshot in enumerate(history):
    print(f"Step {idx}:")
    print(f"Checkpoint ID:{snapshot.config['configurable']['checkpoint_id']}")
    print(f"Node:{snapshot.metadata.get('source')}")
    print(f"messages: { [ m for m in snapshot.values['comments']]}")
    print("")

