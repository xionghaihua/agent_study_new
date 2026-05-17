from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embeddings_model = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
embeddings = embeddings_model.embed_documents(
    [
        "Hi three!",
        "Oh,hello!",
        "What's you name?"
    ]
)
#print(len(embeddings),embeddings)

#句子向量化
embedded_query = embeddings_model.embed_query("what was the name mentioned in conversation?")
#print(embedded_query)
print("===============================================")


