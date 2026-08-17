"""
@author: peter
"""
#==============================================
print("\n============步骤1: 导入依赖于初始化配置=================")
#===============================================
#导入依赖与初始化配置
#步骤1 倒入依赖与初始化配置
import os
import requests
from typing import List,Optional,TypedDict  #用于定义状态类型，保证类型安全
import json
#导入环境变量管理依赖
from dotenv import load_dotenv

#导入langchain相关依赖
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus

#导入langgraph相关依赖
from langgraph.graph import StateGraph,START,END
#导入DashScope相关依赖
import dashscope
try:
    from dashscope.embeddings import TextEmbedding
    from dashscope import Generation
except ImportError:
    raise ImportError("dashscope导入失败，确保已经安装dashscope")
#1.加载环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到DASHSCOPE_API_KEY环境变量，请检查.env文件")
    USE_MOCK = True
else:
    USE_MOCK = False
    print(f"✅ 找到 API key: {api_key[:8]}...")
    #验证API_KEY
    try:
        response = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen3.6-plus",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False
            },
            timeout=10
        )
        if response.status_code == 200:
            print("✅ API key 验证成功")
        elif response.status_code == 401:
            print("❌ API key 无效或无权限")
            USE_MOCK = True
        else:
            print(f"⚠️ API key 验证返回状态码: {response.status_code}")
            print(f"响应: {response.text[:200]}")
    except Exception as e:
        print(f"⚠️ API key 验证异常: {e}")
        USE_MOCK = True

#2 自定义通义前文大模型
class QwenLLM:
    def __init__(
            self,
            model_name:str="qwen3.6-plus",
            api_key:Optional[str]=None,
            use_mock:bool = False,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.use_mock = use_mock
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    def invoke(self,prompt:str,**kwargs)->str:
        if self.use_mock or not self.api_key:
            return f"模拟回答: {prompt[:50]}... (由于无有效API key，使用模拟模式)"
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": self.model_name,
                "messages": [{"role":"user","content":prompt}],
                "temperature": 0.1,
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "无响应内容")
            else:
                error_msg = f"API调用失败:{response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg = error_detail.get("message",error_msg)
                except:
                    error_msg = f"{error_msg}-{response.text[:100]}"
                print(f"❌ {error_msg}")
                return f"调用失败: {error_msg}"
        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            print(f"❌ {error_msg}")
            return f"请求异常: {str(e)[:100]}"
    def __call__(self,prompt:str,**kwargs)->str:
        return self.invoke(prompt,**kwargs)
# 3. 自定义通义千问嵌入模型（正确继承 Embeddings 基类）
class QwenEmbeddings(Embeddings):  #继承Embeddings基类
    """通义千问嵌入模型，继承自 langchain_core.embeddings.Embeddings"""
    def __init__(
        self,
        model_name:str = "text-embedding-v1",
        api_key:Optional[str]=None,
        use_mock:bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.use_mock = use_mock
        if use_mock or not api_key:
            print("🔧 使用模拟嵌入模型")
            self.use_local = True
        else:
            self.use_local = False
    def embed_documents(self,texts:List[str]):
        """嵌入文档文本块"""
        if self.use_local:
            # 返回模拟嵌入向量
            return [[0.1] * 1536 for _ in texts]  # 使用 1536 维度，这是常见嵌入维度
        try:
            import dashscope
            from dashscope.embeddings import TextEmbedding
            dashscope.api_key = self.api_key
            embeddings = []
            for i ,text in enumerate(texts):
                print(f"📄 嵌入第 {i + 1}/{len(texts)} 个文本块...")
                response = TextEmbedding.call(
                    model=self.model_name,
                    input=text[:1000] #限制长度
                )
                if response.status_code == 200 and response.output:
                    embedding = response.output["embeddings"][0]["embedding"]
                    embeddings.append(embedding)
                else:
                    print(f"⚠️ 嵌入失败，使用模拟嵌入: {response.message}")
                    embeddings.append([0.1] * 1536)  # 模拟嵌入
            return embeddings
        except Exception as e:
            print(f"⚠️ 嵌入异常，使用模拟嵌入: {e}")
            return [[0.1] * 1536 for _ in texts]
    def embed_query(self,query:str):
        """嵌入用户查询"""
        if self.use_local or not self.api_key:
            return [0.1] * 1536
        try:
            import dashscope
            from dashscope.embeddings import TextEmbedding
            dashscope.api_key = api_key
            response = TextEmbedding.call(
                model=self.model_name,
                input=query[:1000]
            )
            if response.status_code == 200 and response.output:
                return response.output["embeddings"][0]["embedding"]
            else:
                print(f"⚠️ 查询嵌入失败: {response.message}")
                return [0.1] * 1536
        except Exception as e:
            print(f"⚠️ 查询嵌入异常: {e}")
            return [0.1] * 1536
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入文档"""
        return self.embed_documents(texts)

    async def aembed_query(self, query: str) -> List[float]:
        """异步嵌入查询"""
        return self.embed_query(query)

# 4. 初始化核心组件
llm_model_name = os.getenv("QWEN_MODEL", "qwen3.6-plus")
embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
llm = QwenLLM(
    model_name=llm_model_name,
    api_key=api_key,
    use_mock = USE_MOCK
)
embeddings = QwenEmbeddings(
    model_name=embedding_model_name,
    api_key = api_key,
    use_mock = USE_MOCK
)

# 5. 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)
print("✅ 所有组件初始化成功！")

print("\n======步骤2:构建私有知识库==================")
# 全局变量
vector_db: Optional[Milvus] = None
document_chunks: list[str] = []


from langchain_community.embeddings import FakeEmbeddings
from langchain_milvus import Milvus

def build_knowledge_base(pdf_path:str):
    """
    构建私有知识库：加载PDF文档-->文档分割--->向量嵌入----->存储到FAISS向量库
    :param pdf_path: PDF文档的路径
    :return:  构建好的FAISS向量库对象
    """
    """构建私有知识库"""

    global vector_db, document_chunks
    if not os.path.exists(pdf_path):
        print(f"⚠️ 警告：文件 {pdf_path} 不存在，使用模拟数据")
        document_chunks = [
            "LangChain 是一个用于开发语言模型应用的框架。",
            "LangGraph 是 LangChain 的一个扩展，用于构建有状态的、多智能体的应用程序。",
            "RAG（检索增强生成）是一种结合了信息检索和文本生成的技术。",
            "通义千问是阿里巴巴推出的大型语言模型。",
            "向量数据库用于存储和检索高维向量数据。"
        ]
        # 使用模拟嵌入写入Milvus
        from langchain_community.embeddings import FakeEmbeddings
        fake_embeddings = FakeEmbeddings(size=768)
        vector_db = Milvus.from_texts(
            texts=document_chunks,
            embedding=fake_embeddings,
            connection_args={"uri": "http://172.16.181.128:19530"},
            collection_name="rag_agent",
            drop_old=False,
            auto_id=False,
        )
        return vector_db

    try:
        #加载文档
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ 成功加载 PDF 文档，共 {len(documents)} 页")
        #2.文本分割
        splits = text_splitter.split_documents(documents)
        print(f"✅ 文本分割完成，共得到 {len(splits)} 个文本块")
        document_chunks = [doc.page_content for doc in splits]

        import numpy as np
        from langchain_community.embeddings import FakeEmbeddings
        if USE_MOCK or not api_key:
            print("🔧 使用模拟嵌入构建向量库")
            # 使用随机向量
            fake_embeddings = FakeEmbeddings(size=768)
            embed_model = fake_embeddings
        else:
            print("🔧 使用通义千问嵌入构建Milvus向量库")
            embed_model = embeddings
        #3.写入Milvus

        vector_db = Milvus.from_texts(
            texts=document_chunks,
            embedding=embed_model,
            connection_args={"uri": "http://172.16.181.128:19530"},
            collection_name="rag_agent",
            drop_old=False,  # 生产环境False，不删除历史数据
            auto_id=False,  # 关闭自动ID，如需自定义主键后续自行处理
        )
        print(f"✅ 私有知识库构建完成，数据已存入Milvus collection: rag_agent")
        return vector_db
    except Exception as e:
        print(f"⚠️ 构建知识库失败: {e}")
        print("🔧 使用模拟数据继续...")
        document_chunks = [
            "文档处理失败，使用模拟数据。",
            "错误信息：" + str(e)[:100]
        ]
        from langchain_community.embeddings import FakeEmbeddings
        fake_embeddings = FakeEmbeddings(size=768)
        vector_db = Milvus.from_texts(
            texts=document_chunks,
            embedding=fake_embeddings,
            connection_args={"uri": "http://172.16.181.128:19530"},
            collection_name="rag_agent",
            drop_old=False,
            auto_id=False,
        )
        return vector_db

pdf_path = "./renshi.pdf"
build_knowledge_base(pdf_path)

# def simple_retrieve(query:str,k:int=3) ->List[str]:
#     """简单检索函数"""
#     if vector_db is not None:
#         try:
#             docs = docs = vector_db.similarity_search(query, k=k)
#



"""
build_knowledge_base:封装了知识库构建的完整流程，可直接调用，传入PDF路径
PyPDFLoader: 加载PDF文档
FAISS向量库
检索器: search_kwargs={"k":3},表示返回最相关的3个文本块
"""
#创建检索器
retriever = vector_db.as_retriever(search_kwargs={"k":3})

#步骤3: 定义Langgraph状态
from typing import List,Optional,TypedDict
class AgentState(TypedDict):
    question: str  #用户输入的原始问题
    retrieved_docs: List[str] #从知识库中检索到相应的文本块
    answer:str #智能体生成的最终回答
    need_retrieve: bool  #是否需要检索知识库
#步骤4: 定义langgraph节点
#节点1:判断是否需要检索
def judge_retrieve_node(state:AgentState)->AgentState:
    """
    判断节点：根据用户问题，判断是否需要检索私有知识库
    state：输入的状态
    :return: 修改后的状态
    """
    question = state["question"]
    print(f"\n进入判断节点:{question}")
    prompt = ChatPromptTemplate.from_message([
        ("system",
         "你是一个专业的检索判断助手，仅负责判断用户问题是否需要依赖私有知识库回答："
         "如果问题时通用知识，无需私有知识库即可回答，返回False",
         "如果问题中设计人事，财务的关键字，返回True",
         "注意:仅返回布尔值{True/False),不添加任何额外内容"
         ),
        ("user","用户问题:{question}")
    ])
    judge_chain = prompt | llm | StrOutputParser
    judge_result = judge_chain.invoke({"question":question})
    #将判断结果转唯布尔值，更新状态need_retrieve
    state["need_retrieve"] = judge_result.strip().lower() == "true"
    print(f"判断结果：{state['need_retrieve']}")
    return state

#节点2:执行检索
def retrieve_node(state:AgentState)->AgentState:
    """
    检索节点:从私有知识库检索与用户问题相关的文本块
    :param state:
    :return:
    """
    question = state["question"]
    #调用检索器，执行相似性检索，返回最相关的3个文本块
    retrieved_documents = retriever.invoke(question)
    #提取文本块内容
    retrieved_content = [doc.page_content for doc in retrieved_documents]
    #更新状态中retrieved_docs字段
    state["retrieved_docs"] = retrieved_content
    print(f"检索完成,共找到{len(retrieved_content)}条相关内容")
    for i,content in enumerate(retrieved_content,1):
        print(f"检索结果{1}:{content[:100]}....")
    return state

#节点3:生成最终回答
#该节点的核心功能： 根据用户问题和检索结果，生成准确，简洁的最终回答
def generate_answer_node(state:AgentState)->AgentState:
    """
    生成节点：根据用户问题和检索结果，生成最终回答
    :param state: 输入状态
    :return: 修改后的状态
    """
    #从状态中获取用户问题和检索结果
    question  = state["question"]
    retrieved_docs = state["retrieved_docs"]
    print(f"进入生成节点，开始生成最终回答...")
    if retrieved_docs:
        context = "\n\n".join(retrieved_docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system","你是一个专业的助手，严格根据以下私有知识库的信息回答用户问题"
             "回答要简洁、准确，不要添加无关内容；如果知识库中没有相关信息，直接说'暂无相关信息',不要编造。"
             f"\n\n私有知识库信息:\n{context}"),
            ("user","用户问题:{question}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system","你是一个专业的助手，根据通用知识回答用户问题，回答要简洁，准确，不要添加无关内容"
             "如果不知道答案，直接说'暂无相关信息'."),
            ("user","用户问题:{question}")
        ])
    #构建生成链，提示词-大模型-输出解析
    generate_chain = prompt | llm | StrOutputParser
    #执行生成链
    final_answer = generate_chain.invoke({"question":question})
    #更新状态中answer
    state["answer"] = final_answer
    print(f"回答生成完成")
    return state
"""
三个节点均遵循：输入状态--执行逻辑-修改状态-输出状态
"""