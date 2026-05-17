import chromadb
#持久化存储
client = chromadb.PersistentClient(path="./chroma_db")


#存在这个集合就返回，不存在就创建
collection = client.get_or_create_collection(name="test")
#添加向量
collection.add(
    documents=["Article by john", "Article by Jack", "Article by Jill"],  # 文本内容列表，每个元素是一段文本（如文章、句子等）
    embeddings=[[1,2,3],[4,5,6],[7,8,9]], # 嵌入向量列表，每个元素是一个与 documents 对应的向量表示
    ids=["1","2","3"], ## 自定义 ID 列表，用于唯一标识每条记录
)

#查询数据
aa = collection.get(
    ids=["1"],
    where_document={"$contains":"john"},# 表示文本内容中包含 "john" 的文档
    include=["embeddings"]# 包含嵌入向量, 出于性能考虑，默认不返回嵌入向量
)
print(aa)