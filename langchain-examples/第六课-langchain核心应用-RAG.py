"""
知识点
- 文档加载与分块
- 嵌入模型与向量存储
- 检索器设计
- RAG Agent
- 真实VectorStore与模拟检索的区别
"""
from typing import Callable

from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os

from sqlalchemy.testing.suite.test_reflection import metadata

load_dotenv()
langfuse_handler=CallbackHandler()

#全局向量存储
vector_store={}

def example_1():
    """
    目标：掌握文档处理的基础流程
    知识点：
    - 创建文档对象
    - 文档分块策略
    - chuk_size和chunk_overlap的影响
    """
    print("\n========案例1:文档加载与分块============")
    #模拟文档内容
    raw_documents=[
        Document(
            page_content="""
            langchain是一个用于构建大语言模型应用的开源框架，它提供了丰富的组件和工具，帮助开发者快速构建AI应用，Langchain的核心概念是讲LLM与外部数据源和工具集成。
            主要组件包括：模型接口，提示词模板，索引组件，链式调用和代理
            """,
            metadata={"source":"langchain_intro.txt","section":"简介"},
        ),
        Document(
            page_content="""
            RAG(检索增强生成)是一种结合信息检索和文本生成的技术，它通过检索相关知识库，然后讲检索结果作为上下文提供给LLM。这样可以减少LLM的幻觉，提高回答问题的准确性。RAG的典型应用场景包括：
            智能客服，知识库问答，文档摘要等
            """,
            metadata={"source":"rag_guide.txt","section":"RAG概述"},
        )
    ]
    #文档分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, #每个chunk的字符数
        chunk_overlap=20, #重复的
        #默认分割，separators=["\n\n","\n"," ",""]
        #is_separator_regex=False(默认），默认按普通字符串，若为True表示按正则表达式
    )
    splits = text_splitter.split_documents(raw_documents)
    print(f"原文档数:{len(raw_documents)}")
    print(f"分块后：:{len(splits)}")
    print("\n分块结果：")
    for i,split in enumerate(splits):
        print(f"\n Chunk{i+1}:")
        print(f"内容:{split.page_content}")
        print(f"元数据:{split.metadata}")
    """
            ========案例1:文档加载与分块============
        原文档数:2
        分块后：:4
        
        分块结果：
        
         Chunk1:
        内容:langchain是一个用于构建大语言模型应用的开源框架，它提供了丰富的组件和工具，帮助开发者快速构建AI应用，Langchain的核心概念是讲LLM与外部数据源和工具集成。
        元数据:{'source': 'langchain_intro.txt', 'section': '简介'}
        
         Chunk2:
        内容:主要组件包括：模型接口，提示词模板，索引组件，链式调用和代理
        元数据:{'source': 'langchain_intro.txt', 'section': '简介'}
        
         Chunk3:
        内容:RAG(检索增强生成)是一种结合信息检索和文本生成的技术，它通过检索相关知识库，然后讲检索结果作为上下文提供给LLM。这样可以减少LLM的幻觉，提高回答问题的准确性。RAG的典型应用场景包括：
        元数据:{'source': 'rag_guide.txt', 'section': 'RAG概述'}
        
         Chunk4:
        内容:智能客服，知识库问答，文档摘要等
        元数据:{'source': 'rag_guide.txt', 'section': 'RAG概述'}
    """
def example_2():
    """
    知识点：
    - 文档-嵌入-向量存储-检索-Agent
    """
    print("\n========案例2:真实向量存储-OpenAIEmbeddings+Chroma============")
    #准备文档
    documents = [
        Document(
            page_content="""
            Python是一种流行的编程语言，广泛用于Web开放和数据分析
            """,
            metadata={"topic":"编程"}
        ),
        Document(
            page_content="机器学习是AI的核心技术，包括监督学习，无监督学习和强化学习",
            metadata={"topic":"AI"}
        ),
        Document(
            page_content="Python的Flask框架可以快速构建Rest API",
            metadata={"topic":"编程"}
        ),
        Document(
            page_content="数据库是存储和管理数据的系统，常见的有MySQL，PostgreSQL",
            metadata={"topic":"数据库"}
        )
    ]
    #创建嵌入模型,文档向量化
    embs = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v1"
    )
    # 使用分割器分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    #创建Chroma向量存储
    vectorstore = Chroma.from_documents(docs,embs)  #把内容存到向量数据库
    """
    vectorstore.similarity_search(query,k=4,filter=None,where_document=None)
    query: 搜索关键词
    k：返回最相关的k个文档
    filter: 元数据过滤条件
    where_document：文档内容过滤，如{"$contains":"Python"}
    """
    #基础相似度搜索
    print("\n==相似度搜索（基础）=====")
    results=vectorstore.similarity_search("Python编程",k=2)
    for i,doc in enumerate(results,1):
        print(f"{1},[{doc.metadata.get('topic','未知')}],{doc.page_content}")
    #filter元数据过滤
    print("\n==filter源数据过滤（filter）=====")
    filtered=vectorstore.similarity_search("Python",k=2,filter={"topic":"编程"})
    for doc in filtered:
        print(f"[{doc.metadata['topic']}],{doc.page_content}")
    #where_document内容过滤
    print("\n===文档内容过滤=====")
    content_filtered=vectorstore.similarity_search(
        "编程",
        k=2,
        where_document={"$contains":"Flask"}
    )
    for doc in content_filtered:
        print(f"[{doc.metadata['topic']}],{doc.page_content}")
    print("$not_contains ‘数据库’:")
    not_contains = vectorstore.similarity_search(
        "技术",
        k=2,
        where_document={"$not_contains":"数据库"}
    )
    for doc in not_contains:
        print(f"[{doc.metadata['topic']}],{doc.page_content}")
    """
    as_retriever参数:
    vectorstore.as_retriever(search_type="similarity",search_kwargs={})
    将向量存储包装为langchain Retriever对象
    
    Retriever可以直接将工具传给Agent应用
    
    search_type（检索策略）
    - similarity：默认，普通向量相似度检索，返回最近的k个文档
    - mmr：在相关性和多样性之间平衡，避免返回相似内容
    - similarity_score_threshold 只返回相似度分数超过阈值的结果
    search_kwargs：
    通用参数：
    "k":2,返回文档数
    "filter":{"topic":"编程”} #元数据过滤
    """
    print("\n===检索器：基础模式similarity=====")
    retriever = vectorstore.as_retriever(search_kwargs={"k":2})
    docs = retriever.invoke("Python编程")
    for doc in docs:
        print(f"[{doc.metadata.get("topic")}],{doc.page_content}")

    #mmr模式
    print("\n=====检索器：mmr模式=====")
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":2,"fetch_k":5,"lambda_mult":0.3},
    )
    docs = mmr_retriever.invoke("Python")
    for doc in docs:
        print(f"[{doc.metadata.get("topic")}],{doc.page_content}")
    #score_threshold模式（最低相似度阈值）
    print("\n====检索器：score_threshold模式===")
    threshold_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k":2,"score_threshold":0.6},
    )
    docs = threshold_retriever.invoke("Python")
    if docs:
        for doc in docs:
            print(f"[{doc.metadata.get("topic")}],{doc.page_content}")
    else:
        print("无满足阈值的结果")
    #带filter的检索器
    print("\n=====检索器：元数据过滤=======")
    filtered_retriever = vectorstore.as_retriever(
        search_kwargs={"k":2,"filter":{"topic":"编程"}},
    )
    docs = filtered_retriever.invoke("AI技术")
    for doc in docs:
        print(f"[{doc.metadata.get("topic")}],{doc.page_content}")
    #检索工具
    @tool
    def search_documents(query:str)->str:
        """
        搜索相关文档
        参数：
          query：搜索关键词
        返回：
            相关文档内容
        """
        docs = retriever.invoke(query)
        if not docs:
            return f"未找到关于{query}相关文档"
        output = f"找到{len(docs)}个相关文档\n"
        for i,doc in enumerate(docs,1):
            output += f"[{doc.metadata.get("topic","未知")}],{doc.page_content}\n"
        return output
    #创建agent
    model=init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    agent = create_agent(
        model=model,
        tools=[search_documents],
        system_prompt="""你是一个知识库助手，基于检索到的文档回答问题，如果文档中没有相关信息，请说明“知识库暂无相关信息”。""",
    )
    result1 = agent.invoke(
        {"messages":[("user","Python可以用来做什么")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")
    result2 = agent.invoke(
        {"messages":[("user","机器学习包含哪些技术")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")


def example_3():
    """
    chromax详解：持久化、多聚合
    - 持久化存储
    - 多集合管理
    - 常用方法演示
    :return:
    """

    print("\n=========案例3:Chroma详解-持久化，多集合，常用方法=======")
    embs = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v1"
    )
    # 持久化存储演示
    print("\n=====持久化存储演示=========")
    print("""
    Chroma默认将数据存储在内存中，程序退出后数据丢失，要持久化保存，只需传入persist_directory参数
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name="my_collection",
        embedding_function=embs,
    )
    数据自动保存到./chroma_db/
    磁盘存储结构：
    ./chroma_db/
    - chroma.sqlite3 - SQLite数据库
    - *.bin    - 向量索引文件
    """)

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name="demo_persistent",
        embedding_function=embs,
    )
    #添加文档并持久化
    demo_docs = [
        Document(page_content="持久化数据可以跨会话保存。",metadata={"type":"demo"}),
        Document(page_content="Chroma使用SQLite存储元数据",metadata={"type":"demo"}),
    ]
    vectorstore.add_documents(demo_docs)
    print(f"已添加{len(demo_docs)}条文档到持久化存储")

    #验证：重新查询
    results = vectorstore.similarity_search("数据存储",k=1)
    print(f"查询结果： {results[0].page_content if results else '无结果'}")

    #多集合管理
    """
    一个项目可以有多个向量库，共享同一个底层数据库，但通过collection_name实现数据隔离
    典型场景
    - faq_store 问答知识
    - product_store 产品文档
    - support_store 客服工单知识
    """
    faq_store = Chroma(
       persist_directory="./chroma_db",
       collection_name="faq_store",
       embedding_function=embs,
    )
    product_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="product_store",
        embedding_function=embs,
    )
    faq_docs = [
        Document(page_content="如何重置密码？请在设置页面点击忘记密码。",metadata={"category": "账户"}),
        Document(page_content="如何升级会员？访问官网的订阅页面即可操作",metadata={"category":"付费"}),
        Document(page_content="支持哪些支付方式？支持支付宝、微信、信用卡",metadata={"category":"付费"}),
    ]
    faq_store.add_documents(faq_docs)
    print(f"FAQ集合：已添加{len(faq_docs)}条文档")

    product_docs = [
        Document(page_content="Pro版支持无限协作和高级分析功能",metadata={"product":"Pro"}),
        Document(page_content="基础版免费使用，适合个人开发者",metadata={"product":"Basic"}),
    ]
    product_store.add_documents(product_docs)
    print(f"产品集合：已添加{len(product_docs)}条文档")

    #跨集合检索验证，数据完全隔离
    print("\n====FAQ搜索======")
    faq_results = faq_store.similarity_search("支付",k=2)
    for doc in faq_results:
        print(f"[{doc.metadata.get('category','未知')}],{doc.page_content}")
    print("\n==产品搜索====")
    product_results = product_store.similarity_search("免费版",k=2)
    for doc in product_results:
        print(f"[{doc.metadata.get('product','未知')}],{doc.page_content}")

    """
    Chroma常用方法的演示
    get() 查询全部数据
    count() 查询文档总数
    peek(n)  预览前n条
    similarity_search(query,filter=...)，带过滤的搜索
    add_documents(),添加新文档
    delete(ids=...) 删除指定文档
    
    """
    print("1.get()-查询全部数据")
    all_data = faq_store.get()
    print(f"返回字段：{list(all_data.keys())}")

    print("2.count()查看文档总数")
    count=faq_store._collection.count()
    print(f"FAQ集合共有{count}条文档")

    print("3.similarity_search(query,filter=...)元数据过滤搜索")
    filtered_results = faq_store.similarity_search("方法",k=1,filter={"category":"付费"})
    for doc in filtered_results:
        print(f"----{doc.page_content}----")

    #添加新文档
    """
    不穿入ids（默认）： chroma会自动添加UUID，即使内容完全相同，也会作新文档存储
    传入ids： 使用你提供的ID作为唯一标识，如果ID存在，chroma会覆盖源内容，upsert机制
    
    faq_store.add_documents([new_doc],ids=["custom_id_001"])
    """
    print("4.add_documents()添加新文档")
    new_doc = Document(page_content="如何联系客服？发送邮件到support@163.com",metadata={"category":"账户"})
    faq_store.add_documents([new_doc]) #默认生成UUID
    new_count = faq_store._collection.count()
    print(f"添加后总数：{new_count}条")

    print("5.delete()删除数据")
    temp_store = Chroma(
        persist_directory="./chroma_db",
        collection_name="temp_demo",
        embedding_function=embs,
    )
    temp_store.add_documents(
        [
            Document(page_content="这是一条临时数据",metadata={"temp":True}),
        ]
    )
    print(f"删除前：{temp_store._collection.count()}条")
    temp_data = temp_store.get()
    if temp_data.get("ids"):
        temp_store.delete(ids=[temp_data.get("ids")[0]])
        print(f"删除后：{temp_store._collection.count()}条")

    #数据去重
    """
    chroma本身不自动根据内容去重，它通过ID识别文档
    - 如果添加相同ID的文档，会覆盖源有内容
    - 如果添加不同ID，但内容相通，会覆盖存储
    
    """
    print("策略1:内容哈希作为ID")
    import hashlib
    def make_doc_id(text:str)->str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
    hash_store = Chroma(
        persist_directory="./chroma_db2",
        collection_name="temp_hash",
        embedding_function=embs,
    )
    doc1 = Document(page_content="langchain是一个好框架")
    doc1_id = make_doc_id(doc1.page_content)
    hash_store.add_documents([doc1],ids=[doc1_id])
    print(f"添加doc1（ID:{doc1_id}):{doc1.page_content[:30]}")

    #第二次添加
    doc1_dup = Document(page_content="langchain是一个好框架")
    doc1_dup_id = make_doc_id(doc1_dup.page_content)
    hash_store.add_documents([doc1_dup],ids=[doc1_dup_id])
    print(f"添加重复 doc1_dup (ID:{doc1_dup_id})覆盖原数据")
    print(f"当前总数:{hash_store._collection.count()}条")
    #添加不同内容
    doc2 = Document(page_content="机器学习是AI的核心技术")
    doc2_id = make_doc_id(doc2.page_content)
    hash_store.add_documents([doc2],ids=[doc2_id])
    print(f"添加doc2 （ID：:{doc2_id}）:{doc2.page_content[:30]}....")
    print(f"当前总数:{hash_store._collection.count()}条")




def example_4():
    """
    RAG对话系统集成Chroma和记忆
    - 真实文档嵌入与存储
    - 多轮对话中的检索增强
    - 上下文保持（InMemoryServer)
    """
    print("\n================案例4: RAG对话系统==============")
    from langchain_core.documents import Document
    tech_docs = [
        Document(
            page_content="Langchain是构建LLM应用的框架，提供Agent、Chain、Tool等组件",
            metadata={"topic":"Langchain"},
        ),
        Document(page_content="Agent是能自主决策的AI程序，结合LLM和工具完成任务",metadata={"topic":"Agent"}),
        Document(page_content="RAG是检索增强生成技术，线检索相关知识再生成回答",metadata={"topic":"RAG"}),
        Document(page_content="Tool是Agent可调用的函数，扩展Agent的能力边界",metadata={"topic":"Tool"}),
        Document(page_content="Chain是将多个步骤川联执行的LLM工作流",metadata={"topic":"Chain"}),
    ]
    embs = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v1"
    )
    #创建Chroma向量库
    vector_store=Chroma.from_documents(
        tech_docs,
        embs,
        collection_name="tech_kb_demo"
    )
    #定义检索工具
    @tool
    def retrieve_context(query:str) ->str:
        """
        从技术知识库检索上下文
        参数：
            query: 查询关键词
        返回:
            检索到的相关文档内容
        """
        results = vector_store.similarity_search(query,k=2)
        if not results:
            return "未找到相关信息"
        return "\n".join([f"[{doc.metadata.get('topic')}] {doc.page_content}" for doc in results ])

    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0.3
    )
    """
    InMemorySaver：Langgraph提供的一种内存级检查点机制（checkpointer）
    核心作用： 在多次invoke调用之间，保持对话状态的连续性
    
    工作原理：
    1.状态保存（save state）
    每次Agent完成任务后，InMemorySaver会将当前的状态保存在内存字典中，使用thread_id作为key
    2.状态加载（Load State）
    下一次调用invoke，如果提供相同的thread_id，InMemorySaver会自动从内存中加载当前的状态
    这意味着Agent可以看到之前的对话历史，从而实现多轮对话
    
    生产环境推荐使用PstgreSQL，Redis
    """
    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt="""
        你是一个技术文档助手，使用检索工具获取准确信息后回答用户问题，如果检索不到相关信息，请如实告知.
        """,
        checkpointer=InMemorySaver()
    )
    #多轮对话测试
    thread_config = {"configurable":{"thread_id":"rag_chat_real_1"}}
    print("\n----第一轮对话-----")
    result1 = agent.invoke(
        {"messages":[("user","什么是Langchain")]},
        config={**thread_config,"callbacks":[langfuse_handler]},
    )
    print(f"AI: {result1['messages'][-1].content}")

    print("\n-----第二轮对话-------")
    result2 = agent.invoke(
        {"messages":[("user","它的Agent是什么？")]},
        config={**thread_config,"callbacks":[langfuse_handler]},
    )
    print(f"AI: {result2['messages'][-1].content}")




def main(example_number:int):
    print("="*60)
    print("第4课:提示词和中间件")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(4)