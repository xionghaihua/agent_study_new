from pydantic import BaseModel
class User(BaseModel):
     name:str = ""
     age:int = 0

user01 = User(name="李娜",age=22)
print(user01)


