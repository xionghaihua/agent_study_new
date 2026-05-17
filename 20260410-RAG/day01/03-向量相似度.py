import numpy as np
from numpy import dot
from numpy.linalg import norm
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client=OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),base_url=os.getenv("DASHSCOPE_BASE_URL"))

def cos_sim(a,b):
    #余弦相似度--越大越相似,最大为1
    return dot(a,b)/(norm(a)*norm(b))

def l2(a,b):
    #欧式距离--越小越相似
    x = np.asarray(a) - np.asarray(b)
    return norm(x)

def get_embeddings(texts,model="text-embedding-v1"):
    data = client.embeddings.create(input=texts,model=model).data
    return [x.embedding for x in data]
documents = [
    "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    "我国首次在空间站开展舱外辐射生物学暴露实验",
]
doc_vecs = get_embeddings(documents)
#print(doc_vecs)

query="我国开展舱外辐射生物学暴露实验"
query_vec = get_embeddings(query)
print("余弦相似度:")
for vec in doc_vecs:
    print(cos_sim(query_vec,vec))

print("\n欧式距离:")
for vec in doc_vecs:
    print(l2(query_vec,vec))
