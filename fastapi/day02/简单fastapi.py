#pip install fastapi
#pip install uvicorn
import uvicorn
from fastapi import FastAPI,Path
from typing import Annotated

#创建对象
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

#路径参数
#路径参数 user_id 的值将作为参数 user_id 传递给你的函数
#有类型的路径参数
#你可以使用标准的 Python 类型标注为函数中的路径参数声明类型
@app.get("/user/{user_id}")
def get_user(user_id:int):
    print(user_id, type(user_id))
    return {"user_id": user_id}

from enum import Enum
class ModelName(str, Enum):
    English = "英语"
    Chinese= "中文"
    French = "法语"

@app.get("/models/{model_name}")
async def get_model(model_name:ModelName):
    if model_name is ModelName.English:
        return {"model_name":model_name,"message":"这是英语"}
    if model_name is ModelName.Chinese:
        return {"model_name":model_name,"message":"这是中文"}
    return {"model_name":model_name,"message":"这是法语"}

#Path对路径参数进行限制
@app.get("/users/{id}")
async def get_user(
        id:Annotated[int,Path(description="用户ID",gt=0)]
):
    return {"id": id,"message":f"这是用户id为{id}的人"}

#路径参数中包含文件路径时，比如 /files/{file_path}，我们可以用声明file_path类型的方式进行支持

@app.get("/files/{file_path:path}")
async def read_file(file_path:str):
    return {"file_path":file_path}

#我们可以在路径操作中添加其他参数，比如标签列表
from pydantic import BaseModel
class Item(BaseModel):
    name: str
    description: str
    price: float
    tax: float
@app.get("/items",response_model=Item,tags=["items"])
async def create_item(*,item:Item):
    return {"item":item}

@app.get("/items/",tags=["items"])
async def read_items():
    return [{"name": "Foo", "price": 42}]
@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "johndoe"}]


#枚举限制路径参数
from enum import Enum
class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resNet"
@app.get("/models2/{model_name}")
async def get_model(model_name:ModelName):
    if model_name == ModelName.alexnet:
        return {"model":model_name}
    return {"model":model_name}

#路径参数 + 查询参数混用
from fastapi import Query
@app.get("/users/{user_id}/articles")
def user_articles(
        user_id:int,
        page: int = Query(default=1,description="页面",gt=0,lt=101),
        size:int=10
):
    return {"user_id":user_id,"page":page,"size":size}

from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/items3/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

#Cookie
from fastapi import Cookie
@app.get("/items/cookies/")
async def get_cookie(id:Annotated[str|None,Cookie()] = None):
    return {"id": id}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)