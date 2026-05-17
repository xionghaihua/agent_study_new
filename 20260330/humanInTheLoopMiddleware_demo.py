from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.types import Command
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
load_dotenv()

model = init_chat_model(
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
    temperature=0.1,
    timeout=60,
    max_tokens=2000,
    model_provider="openai"
)
#人工干预中间件
human_in_loop = HumanInTheLoopMiddleware(
    interrupt_on={
        "advanced_search":True # 需要审批，默认允许 approve, edit, reject
    },
    description_prefix="工具执行待审核"
)
@tool
def advanced_search(keyword:str):
    """
       高级搜索功能，根据关键词搜索商品库存。

       参数:
           keyword (str): 搜索关键词，例如 "iPhone 15 Pro"

       返回:
           str: 搜索结果，包含商品名称、库存数量和店铺信息
       """
    print("我被调用啦~~")
    print(f"关键词'{keyword}'")
    return """
       1. [实体店A] iPhone 15 Pro - 库存：8台
       2. [实体店B] iPhone 14 - 库存：12台
       3. [实体店C] iPhone 15 - 库存：6台
       4. [实体店D] iPhone SE (第三代) - 库存：15台
       5. [实体店E] iPhone 15 Pro Max - 库存：400000台
       """
demo_agent=create_agent(
    model=model,
    tools=[advanced_search],
    system_prompt="你是一个专业的商品库存查询助手。",
    checkpointer=InMemorySaver(),
    middleware=[human_in_loop]
)
config = {"configurable": {"thread_id": "1"}}
result = demo_agent.invoke(
    {"messages": [{"role": "user", "content": "请搜索iPhone 15 Pro的库存"}]},
    config=config
)
#
#打印人工审批信息
interrupt_data=result["__interrupt__"]
interrupt_entry = interrupt_data[0]   # 取出唯一的 Interrupt 对象
value = interrupt_entry.value
print("=== 人工审批信息 ===")
print(f"审批ID: {interrupt_entry.id}\n")
for req in value["action_requests"]:
    print(f"工具名称: {req['name']}")
    print(f"调用参数: {req['args']}")
    print(f"描述: {req['description']}\n")

for cfg in value["review_configs"]:
    print(f"针对工具 '{cfg['action_name']}' 允许的决定: {cfg['allowed_decisions']}")
print("===================")

#人工决策并恢复流程
# 等待用户输入决定审批
user_input = input("请输入审批操作（approve / edit / reject）：").strip().lower()
if user_input == "approve":
    result = demo_agent.invoke(
        Command(resume={"decisions":[{"type":"approve"}]}),
        config=config
    )
elif user_input == "reject":
    reason = input("请输入拒绝原因：").strip()
    result = demo_agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": reason
                    }
                ]
            }
        ),
        config=config
    )
elif user_input == "edit":
    new_tool = input("请输入新的工具名（留空保持原工具）：").strip() or "advanced_search"
    new_keyword = input("请输入新的关键词：").strip()
    result = demo_agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": new_tool,
                            "args": {"keyword": new_keyword}
                        }
                    }
                ]
            }
        ),
        config=config
    )
else:
    print("未知操作，默认拒绝")
    result = demo_agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": "未知审批操作，默认拒绝"
                    }
                ]
            }
        ),
        config=config
    )

# -------------------- 8. 输出最终结果 --------------------
print(result["messages"][-1].content)

