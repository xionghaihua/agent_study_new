"""
CharacterTextSplitter 按固定字符数切割
RecursiveCharacterTextSplitter 递归按分割符分割
MarkdownHeaderTextSplitter 按Markdown标记层分割
"""

#RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader('./人事管理文档.pdf')
docs = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
#直接对文档进行分割
# paragraphs = text_splitter.split_documents(docs)
# for i in range(len(paragraphs)):
#     print(f"{i},{len(paragraphs[i].page_content)} {paragraphs[i].page_content}")


#转换成文档对象
paragraphs = text_splitter.create_documents([page.page_content.replace('\n','').replace(' ','') for page in docs if docs ])
#print(paragraphs)


