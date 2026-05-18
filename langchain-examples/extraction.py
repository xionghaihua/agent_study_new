"""
Extraction是从一段文本中解析数据的过程
通常与Extraction parser一起使用，以构建数据
从句子中提取结构化行以插入数据库
从长文档中提取多行以插入数据库
从用户查询中提取参数以进行 API 调用
最近最火的 Extraction 库是 KOR

"""

from dotenv import load_dotenv
import os
from langchain.messages import HumanMessage
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema
load_dotenv()
chat_model = ChatOpenAI(
    temperature=0,
    model="qwen3.5-122b-a10b",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)
instructions = """
You will be given a sentence with fruit names, extract those fruit names and assign an emoji to them
Return the fruit name and emojis in a python dictionary
"""
fruit_names = """
Apple, Pear, this is an kiwi
"""
prompt = (instructions + fruit_names)
output = chat_model.invoke([HumanMessage(content=prompt)])
print (output.content)
print (type(output.content))

#自动格式转换
response_schemas = [
    ResponseSchema(name="artist", description="The name of the musical artist"),
    ResponseSchema(name="song", description="The name of the song that the artist plays")
]
#解析器将会把LLM的输出使用我定义的schema进行解析并返回期待的结构数据给我
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
#print(format_instructions)

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("Given a command from the user, extract the artist and song names \n \
                                                    {format_instructions}\n{user_prompt}")
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions":format_instructions}
)
fruit_query = prompt.format_prompt(user_prompt="I really like So Young by Portugal. The Man")
#print(fruit_query.messages[0].content)

fruit_output = chat_model.invoke(fruit_query.to_messages())
output = output_parser.parse(fruit_output.content)
print(output)