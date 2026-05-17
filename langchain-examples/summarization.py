#文本总结
"""
扔给LLM一段文本，让他给你生成总结可以说是最常见的场景之一了
目前最火的应用应该是 chatPDF
"""
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

model = init_chat_model(
    model_provider="openai",
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.5-122b-a10b",
    temperature=0
)
# 创建模板
template = """
%INSTRUCTIONS:
Please summarize the following piece of text.
Respond in a manner that a 5 year old would understand.
%TEXT:
{text}
"""
#创建一个langchain prompt模版
prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

# 打印模板内容
long_text = """
For the next 130 years, debate raged.
Some scientists called Prototaxites a lichen, others a fungus, and still others clung to the notion that it was some kind of tree.
“The problem is that when you look up close at the anatomy, it’s evocative of a lot of different things, but it’s diagnostic of nothing,” says Boyce, an associate professor in geophysical sciences and the Committee on Evolutionary Biology.
“And it’s so damn big that when whenever someone says it’s something, everyone else’s hackles get up: ‘How could you have a lichen 20 feet tall?’”
"""
final_prompt = prompt.format(text=long_text)
#print(final_prompt)

output = model.invoke(final_prompt)
#print(output)

print("===============================")
#长文本总结
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open('./data/alice_in_wonderland.txt', 'r') as file:
    text = file.read() # 文章本身是爱丽丝梦游仙境
# 安装用于分割文本的依赖
#pip install tiktoken
num_tokens = model.get_num_tokens(text)
#print (f"There are {num_tokens} tokens in your file, file_size: {text.__len__()}")

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"],
    chunk_size=5000,
    chunk_overlap=350
)
docs = text_splitter.create_documents([text])
#print (f"You now have {len(docs)} docs intead of 1 piece of text")

# 使用 map_reduce的chain_type，这样可以将多个文档合并成一个
chain = load_summarize_chain(llm=model,chain_type='map_reduce')
# 典型的map reduce的思路去解决问题，将文章拆分成多个部分，再将多个部分分别进行 summarize，最后再进行 合并，对多个 summary 进行 summary
output = chain.run(docs)
print(output)