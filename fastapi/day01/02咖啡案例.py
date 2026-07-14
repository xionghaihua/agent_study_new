#同步代码
import time
def make_coffee_sync(name,time_cost):
    print(f"店员：开始做{name},需要{time_cost}秒")
    #同步等待
    time.sleep(time_cost)
    print(f"店员：{name}做好了")

def coffee_shop_sync():
    start_time = time.time()
    make_coffee_sync("美式咖啡",3)
    make_coffee_sync("拿铁咖啡",2)
    make_coffee_sync("卡布奇诺",1)
    total_time = time.time() - start_time
    print(f"\n同步模式总耗时:{total_time:.1f}秒")

if __name__ == "__main__":
    print("======同步模式========")
    coffee_shop_sync()

