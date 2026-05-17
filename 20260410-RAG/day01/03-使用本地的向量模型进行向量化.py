#pip install sentence-transformers

from sentence_transformers import SentenceTransformer

model_path = "/Users/mac/Desktop/local_model/BAAI/bge-large-zh-v1.5"
model=SentenceTransformer(model_path)

sentences = [
    "您好",
    "我来自上海"
]
embeddings = model.encode(sentences)
print(embeddings)
print(embeddings.shape)


