#pip install pypdf
##加载离线的pdf
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("人事管理文档.pdf")
pages = loader.load_and_split()
#print(pages[2])


#加载word
#pip install unstructured
#pip install python-doc
#pip install python-docs

#from langchain_community.document_loaders import UnstructuredWordDocumentLoader
#doc = UnstructuredWordDocumentLoader('十五五.docx')
#print(doc.load())


#加载在线的pdf
from langchain_community.document_loaders import PyPDFLoader
loader2 = PyPDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader2.load()
print(data[0].page_content)


