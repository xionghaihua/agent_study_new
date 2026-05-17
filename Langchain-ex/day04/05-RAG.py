from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import DashScopeEmbeddings
#加载本地向量模型
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

#读取文档内容
loader = PyPDFLoader('人事管理文档.pdf')
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
#将数据进行切割成块
paragraphs = text_splitter.create_documents([ page.page.content.replace('\n','').replace(' ','') for page in pages if pages])

#创建嵌入模型
model_name = '/Volumes/DATA/local_model/BAAI/bge-large-zh-v1___5'
embeddings = HuggingFaceEmbeddings(model_name=model_name)
#向量化
db = Chroma.from_documents(paragraphs,embeddings,persist_directory='chroma_db')

#加载本地向量数据库
#db = Chroma(persist_directory='chroma_db',embedding_function=embeddings)

query = "新员工试用期一般为多久"