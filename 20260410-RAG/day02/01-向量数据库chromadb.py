import chromadb
from base_llm import client
import json

class MyVectorDBConnector:
    def __init__(self,collection_name):
        chroma_client = chromadb.PersistentClient(path="../day01/chroma_db")
        #创建一个collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "consine"} #默认为欧式距离，添加这个就是采用余弦相似度
        )
    #文档转化成向量
    def get_embeddings(self,texts,model="text-embedding-v1"):
        """封装Embedding的模型接口"""
        data = client.embeddings.create(input=texts,model=model).data
        return [x.embedding for x in data]
    #文档添加到向量数据库
    def add_document(self,instructions,outputs):
        """向collection中添加文档"""
        embeddings = self.get_embeddings(instructions)
        #将向量化的数据和原文存入向量数据库
        self.collection.add(
            embeddings=embeddings,
            documents=outputs,
            ids=[ f"id{i}" for i in range(len(outputs))]
        )
    #向量数据库搜索
    def search(self,query):
        #把我们查询的问题向量化，在chroma中查询
        results = self.collection.query(
            query_embeddings = self.get_embeddings([query]), #转换成向量
            n_results=2 #匹配多少个数据
        )
        return results
if __name__ == "__main__":
    with open('train_zh.json', 'r', encoding='utf-8') as f:
        data = [ json.loads(line) for line in f]
    #print(data[0:200])
    #获取前10条的问题和输出
    instructions = [ entry['instruction'] for entry in data[0:10]]
    outputs = [ entry['outputs'] for entry in data[0:10]]
    #创建一个向量数据库对象
    vector_db = MyVectorDBConnector("demo")
    #向向量数据库添加文档
    vector_db.add_document(instructions,outputs)
    #print(vector_db.collection.get())
    user_query="得了白癜风怎么办"
    results = vector_db.search(query=user_query)
    print(results)
    for para in results['documents'][0]:
        print(para + "\n")

