from dotenv import load_dotenv
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END  # LangGraph 1.0.5
import os
import requests
load_dotenv()
api_key = os.getenv("DASH_API_KEY")
def call_dashscope(prompt:str)->str:
    """本地模拟 API 响应，非线性流程核心逻辑不变"""
    if "分类" in prompt:
        return "订单问题" if "订单" in prompt or "OD" in prompt else "产品咨询" if "产品" in prompt else "投诉建议"
    elif "提取订单号" in prompt:
        return "OD123456" if "OD123456" in prompt else "无"
    elif "智能音箱" in prompt and "蓝牙" in prompt:
        return "1. 支持蓝牙5.0；2. 手机搜设备配对；3. 兼容主流系统"
    elif "校验" in prompt:
        return "不完整" if "请提供订单号" in prompt else "完整"
    return "系统暂时无法服务"

#状态定义
class SupportState(TypedDict):
    user_query: str
    query_type: Optional[str]
    order_id: Optional[str]
    response: Optional[str]
    retry_count: int
    current_node: str  # 唯一控制字段：指定当前要执行的节点
    is_complete: bool
#核心节点（每个节点仅处理自身逻辑，不依赖边关系分支
def classify_query(state:SupportState)->SupportState:
    """分类节点：仅处理分类，更新下一个节点"""
    print(f"[节点执行] 分类查询：{state['user_query']}")
    query_type = call_dashscope(f"分类用户查询：{state['user_query']}")
    # 非线性分支：根据分类指定下一个节点
    next_node = "extract_order_id" if query_type == "订单问题" else "handle_product" if query_type == "产品咨询" else "handle_complaint"
    return {
        **state,
        "query_type": query_type,
        "current_node": next_node,  # 明确下一个要执行的节点
        "is_complete": False
    }
#提取订单号节点
def extract_order_id(state: SupportState) -> SupportState:
    """提取订单号节点：条件跳转"""
    print(f"[节点执行] 提取订单号")
    order_id = call_dashscope(f"提取订单号：{state['user_query']}")
    next_node = "handle_order" if order_id != "无" else "prompt_order_id"
    return {
        **state,
        "order_id": order_id,
        "current_node": next_node,
        "retry_count": state["retry_count"] + 1
    }
#提示补充订单号节点：触发重试
def prompt_order_id(state: SupportState) -> SupportState:
    """提示补充订单号节点：触发重试"""
    print(f"[节点执行] 提示补充订单号")
    response = "请提供6-12位订单号（如OD123456）"
    # 非线性循环：提示后跳转校验，校验不通过则重试
    return {
        **state,
        "response": response,
        "current_node": "validate_response"
    }
#处理订单节点
def handle_order(state: SupportState) -> SupportState:
    """处理订单节点：生成响应"""
    print(f"[节点执行] 处理订单：{state['order_id']}")
    response = f"订单{state['order_id']}已发货，物流SF123456789，明日送达"
    return {**state, "response": response, "current_node": "validate_response"}

#处理产品咨询
def handle_product(state: SupportState) -> SupportState:
    """处理产品咨询节点：生成响应"""
    print(f"[节点执行] 处理产品咨询")
    response = call_dashscope(f"解答产品咨询：{state['user_query']}")
    return {**state, "response": response, "current_node": "validate_response"}
def handle_complaint(state: SupportState) -> SupportState:
    """处理投诉节点：生成响应"""
    print(f"[节点执行] 处理投诉")
    response = "投诉已记录，24小时内联系你"
    return {**state, "response": response, "current_node": "validate_response"}
def validate_response(state: SupportState) -> SupportState:
    """校验节点：非线性核心（终止/重试）"""
    print(f"[节点执行] 校验响应，重试次数：{state['retry_count']}")
    validation = call_dashscope(f"校验响应：{state['response']}")
    #终止条件：响应完整或重试3次
    if validation == "完整" or state["retry_count"] >= 3:
        return {**state, "is_complete": True, "current_node": "END"}
    # 重试逻辑：根据问题类型跳转回对应节点
    retry_node = "extract_order_id" if state["query_type"] == "订单问题" else "handle_product" if state["query_type"] == "产品咨询" else "handle_complaint"
    return {**state, "current_node": retry_node, "retry_count": state["retry_count"] + 1}

#动态路由节点（核心修复：单一入口，串行执行）
def dynamic_router(state: SupportState) -> SupportState:
    """动态路由：根据 current_node 执行对应节点逻辑，避免多节点并发"""
    current_node = state["current_node"]
    print(f"[路由执行] 跳转至节点：{current_node}")
    # 核心：路由节点内部调用目标节点，而非通过边关系触发
    if current_node == "classify_query":
        return classify_query(state)
    elif current_node == "extract_order_id":
        return extract_order_id(state)
    elif current_node == "prompt_order_id":
        return prompt_order_id(state)
    elif current_node == "handle_order":
        return handle_order(state)
    elif current_node == "handle_product":
        return handle_product(state)
    elif current_node == "handle_complaint":
        return handle_complaint(state)
    elif current_node == "validate_response":
        return validate_response(state)
    elif current_node == "END":
        return {**state, "is_complete": True}
    else:
        return {**state, "current_node": "classify_query"}

graph = StateGraph(SupportState)
# 仅添加 1 个路由节点（所有逻辑都在路由内执行）
graph.add_node("dynamic_router", dynamic_router)
graph.set_entry_point("dynamic_router")
graph.add_edge("dynamic_router", END)
# 编译工作流
app = graph.compile()


def test_workflow(user_query: str):
    print(f"\n{'=' * 70}")
    print(f"处理用户查询：{user_query}")
    print(f"{'=' * 70}")

    initial_state: SupportState = {
        "user_query": user_query,
        "query_type": None,
        "order_id": None,
        "response": None,
        "retry_count": 0,
        "current_node": "classify_query",  # 初始节点
        "is_complete": False
    }

    try:
        # 执行工作流（循环直到流程完成）
        result = initial_state
        while not result["is_complete"]:
            result = app.invoke(result)
            # 避免无限循环（双重保障）
            if result["retry_count"] > 5:
                result["is_complete"] = True
                result["response"] = "系统繁忙，请稍后再试"

        print(f"\n【最终响应】：{result['response']}")
        print(f"【流程状态】：完成，总重试次数：{result['retry_count']}")
    except Exception as e:
        print(f"\n【运行错误】：{str(e)}")
    finally:
        print(f"{'=' * 70}\n")


# 测试核心场景
if __name__ == "__main__":
    #test_workflow("我的订单号是OD123456，请问什么时候发货？")
    test_workflow("请问我的订单什么时候发货？")  # 无订单号，触发重试
    # test_workflow("你们的智能音箱支持蓝牙5.0吗？")  # 产品咨询分支