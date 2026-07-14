import time
import asyncio

async def make_coffee_async(name,time_cost):
    print(f"店员：开始做{name},需要{time_cost}秒")
    await asyncio.sleep(time_cost)
    print(f"店员：{name}做好了")
async def coffee_shop_async():
    start_time = time.time()
    await asyncio.gather(
        make_coffee_async("美式咖啡",3),
        make_coffee_async("拿铁咖啡",2),
        make_coffee_async("卡布奇诺",1),
    )
    total_time = time.time() - start_time
    print(f"\n同步模式总耗时:{total_time:.1f}秒")

if __name__ == "__main__":
    print("======异步模式========")
    asyncio.run(coffee_shop_async())