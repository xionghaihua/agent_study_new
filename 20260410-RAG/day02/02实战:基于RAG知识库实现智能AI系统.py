from dotenv import load_dotenv
from openai import OpenAI
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import chromadb
import os

#切割函数
#按照固定字符切割文档
def sliding_windows_chunks(text,chunk_size,stride):
    return [ text[i:i+chunk_size] for  i in range(0,len(text),stride)]


#读取pdf
def extract_text_from_pdf(filename,page_numbers=None,min_line_length=1):
    '''从PDF文件中按指定页码提取文字'''
    paragraphs = []
    buffer = ''
    full_text = ''
    #提取全部文本
    print(extract_pages(filename)) #返回的是生成器
    for i,page_layout in enumerate(extract_pages(filename)):
        #如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            #检查element是不是文本
            if isinstance(element,LTTextContainer):
                #将换行和空格去掉
                full_text += element.get_text().replace('\n','').replace(" ","")
        if full_text:
            #调用切割函数
            text_chunks = sliding_windows_chunks(full_text,250,100)
            for text in text_chunks:
                paragraphs.append(text)
        return paragraphs


#向量数据库类
class MyVectorDBConnector:
    def __init__(self,collection_name):
        chroma_client = chromadb.PersistentClient(path="../day02/chroma_db2")
        # 创建一个collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 默认为欧式距离，添加这个就是采用余弦相似度
        )
    #使用智普模型进行向量化
    def get_embeddings(self,texts,model="text-embedding-v1"):
        data = client.embeddings.create(input=texts,model=model).data
        return [ x.embedding for x in data ]
    def add_documents(self,documents):
        '''向collection中添加文档与向量'''
        self.collection.add(
            embeddings=self.get_embeddings(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )
    def search(self,query,top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.get_embeddings([query]),
            n_results=top_n
        )
        return results

class RAG_Bot:
    def __init__(self,vector_db,n_results=2):
        self.vector_db = vector_db
        self.n_results = n_results

    #llm模型
    def get_completion(self,prompt,model="qwen3.5-122b-a10b"):
        messages = [{"role":"user","content":prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def chat(self,user_query):
        #检索
        search_results = self.vector_db.search(user_query,self.n_results)
        print('search_results:',search_results)

        #构建prompt
        prompt = prompt_template.replace("__INFO__", "\n".join(search_results['documents'][0])).replace("__QUERY__",user_query)
        print("prompt:",prompt)
        #3.调用大模型
        response = self.get_completion(prompt)
        return response

if __name__=="__main__":
    load_dotenv()
    client=OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),base_url=os.getenv("DASHSCOPE_BASE_URL"))
    prompt_template="""
    你是一个问答机器人。
    你的任务是根据下述给定的已知信息回答用户问题。
    确保你的回复完全依据下述已知信息。不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

    已知信息:
    __INFO__

    用户问：
    __QUERY__

    请用中文回答用户问题。
    """
    #使用示例
    docx_filename= "../../Langchain-ex/day04/人事管理文档.pdf"
    #读取pdf
    paragraphs = extract_text_from_pdf(docx_filename)
    #print(paragraphs)
    # 创建一个向量数据库对象
    vector_db = MyVectorDBConnector("demo")
    # 向向量数据库中添加文档
    vector_db.add_documents(paragraphs)

    # 创建一个RAG机器人
    bot = RAG_Bot(
        vector_db
    )
    user_query = ("薪酬方面的?")
    response = bot.chat(user_query)
    print(response)


































