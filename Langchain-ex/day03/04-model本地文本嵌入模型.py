"""
#下载模型到本地
pip install sentence_transformers

from modelscope import snapshot_download
model_dir = snapshot_download(
    'maidalun/bce-embedding-base_v1',
    cache_dir="/Users/mac/Desktop/local_model"
)
"""
from langchain_huggingface import HuggingFaceEmbeddings
model_name = "/Users/mac/Desktop/local_model/BAAI/bge-large-zh-v1___5"
#生成的嵌入向量将被标准化，有助于向量比较
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
)
res = embeddings.embed_documents(["您好","中国"])
print(res)
