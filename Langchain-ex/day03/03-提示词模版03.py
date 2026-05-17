from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

system_template = "你是一个翻译专家，擅长将{input_language}语言翻译成{output_language}语言"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
#print(system_message_prompt)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

prompt_template = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
prompt = prompt_template.format_prompt(input_language="英文",output_language="中文",text="I love Large Language Model").to_messages()
print("prompt:",prompt)

model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="qwen3.5-122b-a10b"
)

res = model.invoke(prompt)
print(res.content)