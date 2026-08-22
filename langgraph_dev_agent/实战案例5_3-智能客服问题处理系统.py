"""
接收用户咨询:先分类
订单问题：查询订单状态
产品咨询：生成产品说明
投诉建议：记录内容转人工
结果校验
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
load_dotenv()
# 从环境变量读取 API Key，代码中不暴露密钥
llm = ChatOpenAI(
    model="qwen3.7-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从 .env 读取
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0
)

#定义工作流状态
#langgraph通过StateGraph管理状态，状态变了需要明确类型
from typing import TypedDict,Optional,Annotated
from langgraph.graph import StateGraph,START,END,MessagesState
#定义工作流状态结构

class SupportState(TypedDict):
    user_query:str #用户原始查询
    query_type:Optional[str] #问题类型
    order_id: Optional[str] #订单号
    response :Optional[str] #最终响应
    is_complete: bool  #响应是否完整
    retry_count: int  # 添加重试计数器

#节点是工作流的执行单元，每个节点接受State并返回更新的State
#问题分类节点，路由入口
#根据用户查询判断问题类型，为动态路由提供依据
def classify_query(state:SupportState)->SupportState:
    """分类用户查询"""
    user_query  = state["user_query"]
    #调用qwen-plus进行分类
    # 调用Deepseek进行分类
    prompt = f"""
        请将用户查询分类为以下三类之一：订单问题、产品咨询、投诉建议
        分类规则：
        - 订单问题：包含订单号、查询物流、修改订单、取消订单等关键词
        - 产品咨询：包含产品功能、使用方法、规格参数、兼容性等关键词
        - 投诉建议：包含投诉、不满、建议、改进等关键词

        用户查询：{user_query}
        仅返回分类结果，无需额外说明。
    """
    query_type = llm.invoke(prompt).content.strip()
    print(f"问题分类结果：{query_type}")
    # 提取订单号（仅订单问题）
    order_id = None
    if query_type == "订单问题":
        extract_prompt = f"""
              从用户查询中提取订单号（通常为6-12位数字或字母+数字组合），
              若未提及订单号，返回"无"。
              用户查询：{user_query}
              仅返回订单号或"无"。
              """
        order_id = llm.invoke(extract_prompt).content.strip()
        print(f"提取订单号：{order_id}")
    return {
        **state,
        "query_type": query_type,
        "order_id": order_id,
        "is_complete": False,  # 初始响应未完成
        "retry_count": 0  # 初始化重试计数器
    }

#分支处理节点
#订单问题
def handler_order(state:SupportState)->SupportState:
    """处理订单问题"""
    order_id = state["order_id"]
    user_query = state["user_query"]
    retry_count = state.get("retry_count", 0)
    if order_id == "无" or not order_id:
        if retry_count > 0:
            response = "抱歉，我仍然无法找到您的订单号。请确认订单号是否正确，或联系人工客服进一步协助。"
        else:
            response = "为了帮你查询订单状态，请提供你的订单号（6-12位数字或字母+数字组合）。"
    else:
        mock_order_status = f"订单{order_id}当前状态：已发货，物流单号：SF123456789，预计明日送达。"
        response = f"你好！{mock_order_status} 若有其他需求，请随时告知。"

    print(f"订单问题响应：{response}")
    return {**state, "response": response, "retry_count": retry_count + 1}

#处理产品咨询
def handle_product(state:SupportState)->SupportState:
    """处理产品咨询，生成产品说明"""
    user_query = state["user_query"]
    retry_count = state.get("retry_count", 0)
    if retry_count > 1:  # 如果已经重试过，使用更简洁的回复
        prompt = f"""
           用户再次咨询：{user_query}
           请用一句话简明扼要地回答这个问题。
           """
    else:
        prompt = f"""
           作为产品顾问，请详细解答用户的产品咨询，语言通俗易懂，结构清晰。

           用户咨询：{user_query}
           回答要求：
           1. 先明确核心问题
           2. 分点说明（不超过3点）
           3. 结尾提供进一步帮助引导
           """
    response = llm.invoke(prompt).content
    print(f"产品咨询响应：{response}")
    return {**state, "response": response, "retry_count": retry_count + 1}

#投诉建议处理（并行执行)
def handle_complaint_record(state: SupportState) -> SupportState:
    """处理投诉建议：记录投诉内容（并行节点1）"""
    user_query = state["user_query"]
    # 模拟写入数据库
    print(f"已记录投诉建议：{user_query}")
    return {**state,"complaint_record":True}

def handle_complaint_forward(state: SupportState) -> SupportState:
    """处理投诉建议：转人工处理（并行节点2）"""
    user_query = state["user_query"]
    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        response = f"您好，您的投诉已升级处理。我们的高级客服主管将亲自跟进您的问题，并在6小时内联系您。"
    else:
        response = f"你好！你的反馈已收到，我们将在24小时内安排专属客服与你联系。感谢你的支持！"
    print(f"投诉转人工响应：{response}")
    return {**state, "response": response, "retry_count": retry_count + 1}

#结果校验节点
def validate_response(state:SupportState)->SupportState:
    """校验响应是否完整，决定是否要重试"""
    response = state.get("response", "")
    user_query = state["user_query"]
    query_type = state.get("query_type", "")
    retry_count = state.get("retry_count", 0)
    # 对于投诉建议，默认认为已处理
    if query_type == "投诉建议":
        print(f"投诉建议已记录并转人工，跳过校验")
        return {
            **state,
            "is_complete": True
        }
    # 如果重试次数超过2次，强制结束
    if retry_count >= 2:
        print(f"已达到最大重试次数({retry_count})，强制结束流程")
        return {
            **state,
            "is_complete": True
        }
    prompt = f"""
    请判断以下响应是否完整解答了用户的查询：
    1. 若响应直接回答了核心问题，或提供了明确的下一步操作指引（如要求补充信息），则判定为完整，返回"完整"
    2. 若响应模糊、未解答核心问题、遗漏关键信息，则判定为不完整，返回"不完整"

    用户查询：{user_query}
    系统响应：{response}
    仅返回"完整"或"不完整"。
    """
    try:
        validation_result = llm.invoke(prompt).content.strip()
        print(f"响应校验结果：{validation_result}")
    except Exception as e:
        print(f"校验过程出错，自动判定为完整：{e}")
        validation_result = "完整"

    return {
        **state,
        "is_complete": validation_result == "完整"
    }

# 初始化状态图
graph = StateGraph(SupportState)

# 添加核心节点
graph.add_node("classify_query", classify_query)  # 问题分类
graph.add_node("handle_order", handler_order)  # 订单处理
graph.add_node("handle_product", handle_product)  # 产品咨询处理
graph.add_node("handle_complaint_record", handle_complaint_record)  # 投诉记录
graph.add_node("handle_complaint_forward", handle_complaint_forward)  # 投诉转人工
graph.add_node("validate_response", validate_response)  # 响应校验

# 从分类节点根据query_type分流到对应处理节点
def route_after_classify(state: SupportState) -> str:
    query_type = state.get("query_type", "")
    if query_type == "订单问题":
        return "handle_order"
    elif query_type == "产品咨询":
        return "handle_product"
    elif query_type == "投诉建议":
        return "handle_complaint_record"  # 先执行记录，再并行转人工
    else:
        return "handle_product"  # 默认走产品咨询

# 添加条件边：分类节点 -> 处理节点
graph.add_conditional_edges(
    "classify_query",
    route_after_classify
)
# 投诉记录后，指向转人工节点
graph.add_edge("handle_complaint_record", "handle_complaint_forward")
# 所有处理节点都指向校验节点
graph.add_edge("handle_order", "validate_response")
graph.add_edge("handle_product", "validate_response")
graph.add_edge("handle_complaint_forward", "validate_response")
# 校验节点的动态路由：完整则结束，不完整则重试对应处理节点
def route_after_validation(state: SupportState) -> str:
    if state.get("is_complete", False):
        return END
    else:
        # 不完整则重试原处理节点
        query_type = state.get("query_type", "")
        if query_type == "订单问题":
            return "handle_order"
        elif query_type == "产品咨询":
            return "handle_product"
        elif query_type == "投诉建议":
            return "handle_complaint_forward"
        else:
            return "handle_product"
# 添加条件边：校验节点 -> 结束或重试
graph.add_conditional_edges(
    "validate_response",
    route_after_validation
)

graph.set_entry_point("classify_query")

# 编译图（生成可执行工作流）
app = graph.compile()

# 测试运行：输入不同类型的用户查询
def test_workflow(user_query: str):
    print(f"\n{'=' * 60}")
    print(f"处理用户查询：{user_query}")
    print(f"{'=' * 60}")
    # 运行工作流
    result = app.invoke({
        "user_query": user_query,
        "query_type": None,
        "order_id": None,
        "response": None,
        "is_complete": False,
        "retry_count": 0
    })
    print(f"\n最终响应：{result.get('response')}")
    print(f"重试次数：{result.get('retry_count', 0)}")
    print(f"{'=' * 60}")
    return result


# 测试函数，运行所有测试用例
def run_all_tests():
    """运行所有测试用例"""
    test_cases = [
        ("我的订单号是OD123456，请问什么时候发货？", "订单问题"),
        ("你们的智能音箱支持蓝牙5.0吗？怎么连接手机？", "产品咨询"),
        ("我对昨天收到的商品不满意，质量有问题，希望退款或换货。", "投诉建议"),
        ("我的订单什么时候能到？", "订单问题但没提供订单号"),
        ("我想问一下这个产品怎么用", "简短的产品咨询")
    ]

    results = []
    for i, (query, description) in enumerate(test_cases, 1):
        print(f"\n{'#' * 60}")
        print(f"测试 {i}: {description}")
        print(f"查询内容: {query}")
        print(f"{'#' * 60}")

        try:
            result = test_workflow(query)
            results.append((query, description, result))
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"所有测试完成，共运行 {len(results)} 个测试用例")
    print(f"{'=' * 60}")


# 测试用例
if __name__ == "__main__":
    run_all_tests()
    # 也可以单独测试
    # test_workflow("我的订单号是OD123456，请问什么时候发货？")
    # test_workflow("你们的智能音箱支持蓝牙5.0吗？怎么连接手机？")
    # test_workflow("我对昨天收到的商品不满意，质量有问题，希望退款或换货。")
    # test_workflow("我的订单什么时候能到？")
    # test_workflow("我想问一下这个产品怎么用")