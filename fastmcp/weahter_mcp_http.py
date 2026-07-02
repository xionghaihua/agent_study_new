from fastmcp import FastMCP

mcp = FastMCP("WeatherService")

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