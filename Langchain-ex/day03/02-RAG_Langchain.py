#加载网页资源
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings  #千问向量模型
from langchain_community.vectorstores import FAISS  #向量数据库
from dotenv import load_dotenv
import os
import bs4
load_dotenv()

"""
vect = None
for i in range(0,len(docs),batch_size=10):
    batch_docs = docs[i:i+batch_size]
    print(f"第{i // batch_size + 1}批次文档数量:{len(batch_docs)}")
    if i==0:
        #第一次
        vect=FAISS.from_documents(batch_docs,embs) 
    else:
        new_vect = FAISS.from_documents(batch_docs,embs)
        vect.merge_from(new_vect)
"""



def faiss_conn():
    # 读取网页中的数据
    loader = WebBaseLoader("https://www.gov.cn/zhengce/content/202510/content_7043916.htm",
                           bs_kwargs=dict(parse_only=bs4.SoupStrainer(id='UCAP-CONTENT')))
    #读取数据
    docs = loader.load()
    #创建向量模型
    embs = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), model="text-embedding-v1")
    #使用分割器分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    #向量存储
    vector = FAISS.from_documents(documents,embs)
    return vector

vector = faiss_conn()
vector.save_local("faiss_save")


#第二步：构建检索
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

embs = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), model="text-embedding-v1")

#检索
save_path = "faiss_save"

vector_store = FAISS.load_local(
    folder_path=save_path,
    embeddings=embs,
    allow_dangerous_deserialization=True, #允许加载pickle文件
)

#创建提示词模版
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题:

<context>
{context}
</context>

问题: {input}""")


llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.5-122b-a10b",
    temperature=0.2
)
#把拼接好的提示词给大模型
doc_chain = create_stuff_documents_chain(llm,prompt)

#创建检索器，把检索的功能进行封装
retriever = vector_store.as_retriever()
retriever.search_kwargs = {"k": 3} #限制为最多检索3个文档

#把检索到的数据，给到doc_chain
res_chain = create_retrieval_chain(retriever,doc_chain)

#执行
res = res_chain.invoke({"input": "密云水库水源保护条例什么时候执行"})
print(res["answer"])

