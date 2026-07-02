from fastmcp import FastMCP
import random

from sympy import resultant

mcp = FastMCP("WeatherService")

#枚举类型
from enum import Enum
class WeatherType(Enum):
    SUNNY = "晴天"
    CLOUDY = "多云"
    RAIN = "下雨"
    SNOW = "下雪"

from pydantic import BaseModel
from datetime import date,datetime,timedelta
from typing import Optional,Union,Literal
from pathlib import Path
class Location(BaseModel):
    province:str
    city:str
    longitude:float
    latitude:float
class WeatherQueryReq(BaseModel):
    location:Location
    query_date:date
    forecast_days:int
    expect_weather: Optional[WeatherType] = None

@mcp.tool(description="基础类型演示：查询多城市实时气温")
def batch_get_temp(city_list: list[str],threshold:float) ->dict[str, float|None]:
    """
    批量查询城市气温，低于阈值返回None
    :param city_list:
    :param threshold:
    :return:
    """
    result={}
    for city in city_list:
        if "city" == "上海":
            result[city] = 28.5
        elif city == "北京":
            result[city] = 35.5
        else:
            result[city] = None
    return result
@mcp.tool(description="传入城市ID或城市名称查询湿度")
def get_humidity(city_key:Union[int,str],default_hum:Optional[float]=50) -> float:
    if isinstance(city_key,int):
        return default_hum + city_key % 20
    return default_hum + len(city_key)
# ===================== 工具3：Literal约束类型 =====================
@mcp.tool(description="Literal限定只能查询今天/明天/后天天气")
def get_fixed_weather(day: Literal["今天", "明天", "后天"]) -> str:
    weather_map = {
        "今天": "晴 27℃",
        "明天": "多云 25℃",
        "后天": "小雨 22℃"
    }
    return weather_map[day]
# ===================== 工具5：Path 文件路径类型 =====================
@mcp.tool(description="Path自动类型转换，读取天气记录文件路径信息")
def get_weather_file_info(file_path: Path) -> dict:
    return {
        "filename": file_path.name,
        "parent_dir": str(file_path.parent),
        "is_absolute": file_path.is_absolute()
    }
# ===================== 工具8：Pydantic结构化模型入参 =====================
@mcp.tool(description="复杂Pydantic嵌套模型查询天气预报")
def forecast_weather(req: WeatherQueryReq) -> dict:
    return {
        "province": req.location.province,
        "city": req.location.city,
        "query_date": req.query_date.isoformat(),
        "forecast_days": req.forecast_days,
        "expect_weather": req.expect_weather.value if req.expect_weather else "不限"
    }

if __name__ == "__main__":
    # HTTP SSE 服务 0.0.0.0:9000
    mcp.run(host="0.0.0.0", port=9000,transport="http")

weather_data = {
    "北京": {"temp": 26, "condition": "晴", "humidity": 35},
    "上海": {"temp": 29, "condition": "多云", "humidity": 62},
    "广州": {"temp": 33, "condition": "雷阵雨", "humidity": 78},
    "深圳": {"temp": 32, "condition": "阴", "humidity": 75},
    "杭州": {"temp": 27, "condition": "小雨", "humidity": 68}
}
@mcp.tool(description="根据城市名称查询当前实时天气")
def get_weather(city:str) -> dict:
    """
    查询指定城市的天气信息
    :param city:  城市中文名，如上海，北京
    :return:  温度，天气状况，湿度
    """
    if city in weather_data:
        return weather_data[city]
    # 未收录城市返回随机模拟天气
    return {
        "temp": random.randint(20, 36),
        "condition": random.choice(["晴", "多云", "小雨", "阴天"]),
        "humidity": random.randint(30, 80)
    }
@mcp.tool(description="获取支持查询的城市列表")
def get_support_city() ->list:
    """返回内置支持精准查询的城市名称列表"""
    return list(weather_data.keys())

if __name__ == "__main__":
    mcp.run(host="0.0.0.0", port=9000,transport="http")