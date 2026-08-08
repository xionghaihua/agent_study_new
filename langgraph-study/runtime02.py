from langgraph.graph import StateGraph,START,END
from langgraph.runtime import Runtime
from typing import TypedDict
from langgraph.graph import MessagesState



class MyContext(TypedDict):
    model:str

MODELS = {
    "anthropic": "anthropic:claude-3.5-haiku-latest",
    "openai": "openai:gpt-4.1-mini"
}
def call_model(state:MessagesState,runtime:Runtime[MyContext]):
    model=""
    if runtime.context:
        model = runtime.context['model']
        model = MODELS[model]
    return {"messages":{"role":"assistant","content":model}}

builder = StateGraph(MessagesState,context_schema=MyContext)
builder.add_node("model",call_model)
builder.add_edge(START,"model")
builder.add_edge("model",END)

graph = builder.compile()

#问题
input_message = {"role":"user","content":"hi"}  #没有配置，使用默认的
response = graph.invoke({"messages":[input_message]})
print(response)

from langgraph.
context = {"model":"openai"}
response_2 = graph.invoke({"messages":[input_message]},context=context)
print(response_2)


#限流
#result = graph.invoke({"messages":[input_message]},context=context,{"recursion_limit":10})