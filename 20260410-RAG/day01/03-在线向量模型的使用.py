import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

#访问模型的厂商
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

#数据封装为向量
def get_embedding(text):
    data = client.embeddings.create(input=text,model="text-embedding-v1")
    return [ i for i in data ]

test_query = [ "我在上海" ]
vec = get_embedding(test_query)
print(vec)