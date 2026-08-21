from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


messages = [
    SystemMessage("你是电商客服助手。订单物流必须基于工具结果回答。"),
    HumanMessage("订单20260001到哪了？"),
    AIMessage(
        content=[],
        tool_calls=[
            {
                "name": "get_order_logistics",
                "args": {"order_no": "20260001"},
                "id": "call_001",
            }
        ],
    ),
    ToolMessage(
        content="订单已发货，物流单号 SF123456，预计明天送达。",
        tool_call_id="call_001",
        name="get_order_logistics",
        artifact={"carrier": "SF", "tracking_no": "SF123456"},
    ),
    AIMessage("我查到订单20260001已发货，物流单号 SF123456，预计明天送达。"),
]

for message in messages:
    print(type(message).__name__, "=>", message.content)