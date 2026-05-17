"""
文档问答(QA over Documents)
为了确保LLM能够执行QA任务

需要向LLM传递能够让他参考的上下文信息
需要向LLM准确地传达我们的问题

"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(
    model_provider="openai",
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
    temperature=0
)
context = """
Rachel is 30 years old
Bob is 45 years old
Kevin is 65 years old
"""

question = "Who is under 40 years old?"
result = model.invoke(context + question)
#print(result.content) #Rachel is under 40 years old.

#长文本
#对于更长的文本，可以文本进行分块，对分块的内容进行 embedding，将 embedding 存储到数据库中，然后进行查询
"""
文档加载（Document Loading）：文档加载器把文档加载为 LangChain 能够读取的形式。有不同类型的加载器来加载不同数据源的数据，如CSVLoader、PyPDFLoader、Docx2txtLoader、TextLoader等。
文本分割（Splitting）：文本分割器把 Documents 切分为指定大小的分割，分割后的文本称为“文档块”或者“文档片”。（本次忽略）
向量存储（Vector Storage）：将上一步中分割好的“文档块”以“嵌入”（Embedding）的形式存储到向量数据库（Vector DB）中，形成一个个的“嵌入片”。
检索（Retrieval）：应用程序从存储中检索分割后的文档（例如通过比较余弦相似度，找到与输入问题类似的嵌入片）。
输出（Output）：把问题和相似的嵌入片（文本形式）都放到提示传递给语言模型（LLM），让大语言模型生成答案。
"""
#pip install faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

loader = TextLoader("./data/wonderland.txt")
doc = loader.load()
#print (f"You have {len(doc[0].page_content)} characters in that document")

# 将小说分割成多个部分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400
)
docs = text_splitter.split_documents(doc)
#向量化
embs=DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="text-embedding-v1"
)
#存储
vectorstore = FAISS.from_documents(docs,embs)
vectorstore.save_local("./data/faiss_save")

#读取
save_path = "./data/faiss_save"
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
query = "What does the author describe the Alice following with?"
retriever = vector_store.as_retriever()
retriever.search_kwargs = {"k": 3} #限制为最多检索3个文档
doc_chain = create_stuff_documents_chain(model,prompt)
#把检索到的数据，给到doc_chain
res_chain = create_retrieval_chain(retriever,doc_chain)

#执行
res = res_chain.invoke({"input": query})
print(res["answer"])
