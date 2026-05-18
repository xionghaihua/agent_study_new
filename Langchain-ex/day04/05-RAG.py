from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

#读取文档内容
loader = PyPDFLoader('人事管理文档.pdf')
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)
#将数据进行切割成块
paragraphs = text_splitter.create_documents([ page.page_content.replace('\n','').replace(' ','') for page in pages if pages])

#创建嵌入模型
embeddings = DashScopeEmbeddings(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), model="text-embedding-v1")
#向量化
#db = Chroma.from_documents(paragraphs,embeddings,persist_directory='chroma_db')

#加载本地向量数据库
db = Chroma(persist_directory='chroma_db',embedding_function=embeddings)

query = "本公司每日工作时间多久"
# docs = db.similarity_search(query)
# print(docs[0].page_content)


#检索器
#用于从文档集合中检索最相关文档
#最基本的检索器,db.as_retriever()
#ParentDocument 父子检索器

