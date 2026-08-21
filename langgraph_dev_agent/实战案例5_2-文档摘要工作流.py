from http.client import responses
from typing import TypedDict,Optional
from dotenv import load_dotenv
import os
import requests
import json
from langgraph.graph import StateGraph,START,END

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

#检验api_key是否存在
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY environment variable not set")
#定义工作流状态结构
class SummaryState(TypedDict):
    raw_document:str
    processed_text: Optional[str]
    summary: Optional[str]
#直接实现DASHSCOPE_API_KEY调用
def call_llm(user_content:str,model: str = "qwen3.8-max",temperature:float=0.2,max_tokens:int=2000)->str:
    """直接调用大模型，不依赖langchain任何模块"""
    url = "https://llm-n99bbo8kcv58e6in.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
    }
    #完整的API请求体
    data = {
        "model": model,
        "messages":[
            {
                "role":"system",
                "content":"你是一个专业的文本摘要助手，请根据用户提供的文本，生成简洁、准确、全面的摘要。摘要长度控制在300字以内，保留原文的核心观点和关键信息。"
            },
            {
                "role":"user",
                "content": user_content
            }
        ],
        "temperature":temperature,
        "max_tokens":max_tokens,
        "stream": False,
         "stop":None
    }
    try:
        #发送API请求
        response = requests.post(url,headers=headers,json=data,timeout=60)
        response.raise_for_status()  #刨除HTTP错误
        result = response.json()
        #调试：保存API的响应
        with open("api_response_debug.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 检查API响应结构
        if "output" in result and "choices" in result["output"]:
            choices = result["output"]["choices"]
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                summary = message.get("content", "").strip()
                if summary:
                    return summary
        # 备用解析逻辑
        if "output" in result and "text" in result["output"]:
            return result["output"]["text"].strip()

        # 如果以上都不存在，尝试其他可能的结构
        print(f"⚠️ API响应格式与预期不符: {result.keys()}")
        return json.dumps(result, ensure_ascii=False, indent=2)[:500]
    except requests.exceptions.Timeout:
        return "LLM调用超时，请检查网络连接或重试"
    except requests.exceptions.ConnectionError:
        return "网络连接失败，请确保能访问"
    except requests.exceptions.HTTPError as e:
        error_detail = f"状态码: {response.status_code}"
        try:
            error_json = response.json()
            error_detail += f", 错误信息: {json.dumps(error_json, ensure_ascii=False)}"
        except:
            error_detail += f", 响应: {response.text[:200]}"
        return f"❌ API请求错误: {error_detail}"
    except Exception as e:
        return f"❌ LLM调用失败: {str(e)[:200]}"
#工作流节点逻辑
def process_document(state:SummaryState)->SummaryState:
    """节点1:文档预处理"""
    raw_doc = state["raw_document"]
    # 增强清洗：处理换行、空格、制表符、全角空格
    processed = (raw_doc.strip()
                 .replace("\n\n", "\n")
                 .replace("  ", " ")
                 .replace("\t", " ")
                 .replace("　", " ")  # 中文全角空格
                 .replace("\r", ""))  # 回车符
    return {"processed_text": processed}
def generate_summary(state:SummaryState)->SummaryState:
    """节点2:调用模型生成摘要"""
    try:
        processed_text = state["processed_text"]
        if not processed_text or len(processed_text) < 10:
            raise ValueError("预处理后的文本为空或过短")
        summary = call_llm(user_content=processed_text)
        return {"summary": summary}
    except Exception as e:
        error_msg = f"❌ 摘要生成失败：{str(e)}"
        print(error_msg)
        return {"summary": error_msg}
def format_summary(state:SummaryState)->SummaryState:
    """节点3:摘要后处理，格式化"""
    summary = state["summary"]
    line_break = "\n"
    # 中英文句号分句，避免格式混乱
    split_summary = summary.replace(". ", f".{line_break}").replace("。 ", f"。{line_break}")
    # 最终格式化（添加标题，去除末尾多余换行）
    formatted_summary = f"### 文档摘要{line_break}{line_break}{split_summary}".rstrip(line_break)
    return {"summary": formatted_summary}

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
config = {"configurable":{"thread_id":"summary_02"}}
# 构建LangGraph线性工作流（核心逻辑不变）
def build_summary_graph():
    """构建并返回工作流实例（仅依赖LangGraph核心）"""
    graph_builder = StateGraph(SummaryState)

    # 添加节点
    graph_builder.add_node("process_doc", process_document)  # 预处理
    graph_builder.add_node("generate_summary", generate_summary)  # 生成摘要
    graph_builder.add_node("format_summary", format_summary)  # 格式优化

    # 定义线性执行顺序：预处理 → 生成摘要 → 格式优化 → 结束
    graph_builder.add_edge("process_doc", "generate_summary")
    graph_builder.add_edge("generate_summary", "format_summary")
    graph_builder.add_edge("format_summary", END)

    # 设置入口节点
    graph_builder.set_entry_point("process_doc")

    return graph_builder.compile(checkpointer=checkpointer)

#可视化工作流（可选，失败不影响核心功能）
def visualize_graph(graph, save_path: str = "summary_workflow.png"):
    """简化可视化逻辑，避免依赖报错"""
    try:
        graph.draw(save_path, format="png")
        print(f"\n✅ 工作流可视化图已保存至：{os.path.abspath(save_path)}")
    except Exception:
        print("\nℹ️  可视化功能未启用（需安装pygraphviz和Graphviz软件，不影响核心功能）")


if __name__ == "__main__":
    # 测试用原始文档（可替换为任意文本）
    raw_document = """
    LangGraph 是 LangChain 生态系统中的一个框架，专门用于构建有状态、可循环的工作流。它基于状态机的思想，允许开发者定义节点和边，
    并通过状态对象管理整个工作流的数据流转。与传统的线性脚本相比，LangGraph 提供了更好的可扩展性和可观测性，
    特别适合 LLM 应用中的复杂流程编排，例如多轮对话、文档分析、工具调用链等场景。LangGraph 的核心组件包括 StateGraph、State、Node 和 Edge，
    这些组件共同构成了灵活且强大的工作流系统。此外，LangGraph 还支持与 LangChain 生态的其他工具无缝集成，
    如提示词模板、向量数据库、工具调用等，进一步降低了复杂 LLM 应用的开发门槛。
    """

    # 初始化工作流
    print("🚀 正在初始化LangGraph线性工作流...")
    summary_graph = build_summary_graph()
    # 可视化工作流（可选）
    visualize_graph(summary_graph)

    # 执行工作流
    print("\n🔄 正在执行文档摘要工作流...")
    result = summary_graph.invoke({
        "raw_document": raw_document
    },config=config)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("📄 最终摘要结果：")
    print("=" * 60)
    print(result["summary"])
    print("=" * 60)