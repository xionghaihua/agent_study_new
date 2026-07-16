"""
目标：掌握跨会话的长期记忆实现
知识点：
- 持久化存储
- 用户画像构建
- 记忆检索与更新
- Profile vs Collection模式
"""
from langchain.chat_models import init_chat_model
from langfuse.langchain import CallbackHandler
from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from langgraph.store.postgres import PostgresStore
import psycopg
from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()
langfuse_handler = CallbackHandler()

DB_URI = "postgresql://admin:admin123@172.16.181.128:5435/ai_memory"
def get_postgres_store():
    """
    获取PostGresSQL存储示例
    直接用psycopy.Connection + PostgresStore构造函数
    内部自动管理JSONB序列化
    注意：PostgresStore要求所有value必需为Dict类型
    :return:
    """
    from psycopg_pool import ConnectionPool
    from psycopg.rows import dict_row
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold":0,
        "row_factory":dict_row,
    }
    pool = ConnectionPool(
        conninfo=DB_URI,
        max_size=5,
        kwargs=connection_kwargs,
    )
    #长期记忆：PostgresStore
    store = PostgresStore(pool)
    store.setup()
    return store

def example_1():
    print("\n==========示例1:基础长期记忆，用户信息存储==========")
    store = get_postgres_store()
    @dataclass
    class UserContext:
        user_id:str
    @tool
    def save_profile(name:str,age:int,occupation:str,runtime:ToolRuntime[UserContext])->str:
        """
        保存用户个人信息，信息将长期保存，跨会话可用
        参数：
          name: 姓名
          age: 年龄
          occupation: 职业
        """
        user_id = runtime.context.user_id
        profile = {"name":name, "age":age, "occupation":occupation}
        store.put(("profiles",),user_id, profile)
        return f"个人信息已保存\n姓名:{name}\n年龄:{age}\n职业：:{occupation}"
    @tool
    def get_profile(runtime:ToolRuntime[UserContext])->str:
        """获取保存的个人信息"""
        user_id = runtime.context.user_id
        profile = store.get(("profiles",),user_id)
        if profile:
            p = profile.value
            return f"个人信息：\n-姓名:{p["name"]}\n-年龄:{p["age"]}\n-职业:{p["occupation"]}"
        return "暂无个人信息，请先保存资料"
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model,
        tools=[save_profile,get_profile],
        system_prompt="你是一个个人信息管理助手",
        context_schema=UserContext,
        store=store,
    )
    print("\n=====测试1: 保存信息=======")
    result1 = agent.invoke(
       {"messages":[("user","我叫张三，今年28岁，是一名程序员")]},
       config={"configurable":{"thread_id":"session_001"},"callbacks":[langfuse_handler]},
       context=UserContext(user_id="U001"),
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n=====测试2: 查询信息=======")
    result2 = agent.invoke(
        {"messages":[("user","我的个人信息是什么")]},
        #长期记忆，可以不用写thread_id
        config={"configurable":{"thread_id":"session_001"},"callbacks":[langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI:{result2['messages'][-1].content}")

def example_2():
    """
    目标：短期记忆和长期记忆如何配合
    知识点：
    - checkpointer 维持当前对话流（短期）
    - Store 保存用户长期画像（长期）
    - 两种同时传入create_agent
    """
    print("\n======示例2:Checkpointer+store组合使用==========")
    store = get_postgres_store()
    #短期
    from langgraph.checkpoint.postgres import PostgresSaver
    check_conn = psycopg.connect(DB_URI,autocommit=True,prepare_threshold=0)
    checkpointer = PostgresSaver(check_conn)
    checkpointer.setup()
    @dataclass
    class UserContext:
        user_id:str
    @tool
    def add_interest_tag(tag:str,runtime:ToolRuntime[UserContext])->str:
        """
        添加用户兴趣标签到Store（长期记忆）
        参数：
            tag：兴趣标签
        """
        user_id = runtime.context.user_id
        data = store.get(("interests",),user_id)
        tags = data.value.get("tags",[]) if data else []
        if tag not in tags:
            tags.append(tag)
        store.put(("interests",),user_id, {"tags":tags})
    @tool
    def get_user_persona(runtime:ToolRuntime[UserContext])->str:
        """从store读取用户画像（store）"""
        user_id = runtime.context.user_id
        interests = store.get(("interests",),user_id)
        tags = interests.value.get("tags",[]) if interests else []
        persona = f"用户画像:(ID:{user_id}):\n"
        persona += f"兴趣标签:{','.join(tags) if tags else '暂无'}\n"
        if "编程" in tags and "AI" in tags:
            persona += "分析:技术型人才，对AI和编程感兴趣"
        elif "编程" in tags:
            persona += "分析：编程爱好者"
        return persona
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model,
        tools=[add_interest_tag,get_user_persona],
        system_prompt="你是一个用户画像分析助手",
        context_schema=UserContext,
        store=store, #长期
        checkpointer=checkpointer, #短期
    )
    print("\n==========1.添加第一个兴趣==========")
    result1 = agent.invoke(
        {"messages":[("user","我对编程感兴趣")]},
        config={"configurable":{"thread_id":"persona_001"},"callbacks":[langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI:{result1['messages'][-1].content}")

    print("\n=====2.chentpointer记住上一轮===========")
    result2 = agent.invoke(
        {"messages":[("user","我还喜欢AI")]},
        config={"configurable":{"thread_id":"persona_001"},"callbacks":[langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI:{result2['messages'][-1].content}")
    print("\n======3.读取画像，store提供长期记忆=======")
    result3 = agent.invoke(
        {"messages":[("user","生成我的用户画像")]},
        config={"configurable":{"thread_id":"persona_001"},"callbacks":[langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI:{result3['messages'][-1].content}")

    print("\n======4.checkpointer重置，但是store数据还存在=======")
    result4 = agent.invoke(
        {"messages":[("user","生成我的用户画像")]},
        config={"configurable":{"thread_id":"persona_002"},"callbacks":[langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI:{result4['messages'][-1].content}")
    print("\n说明：新会话虽然不记得上次对话内容，但store记住了你的长期兴起")


def example_3():
    """
    目标：长期记忆的检索能力
    - 全文检索
    - 按条件检索
    - 相关性排序
    """
    print("\n======示例3:记忆检索与搜索======")
    store = get_postgres_store()
    store.put(
        ("notes","datas"),
        "note_1",
        {"title":"python装饰器","content":"装饰器用于修改函数行为","tags":["python","编程"]},
    )
    store.put(
        ("notes","datas"),
        "note_2",
        {"title":"机器学习基础","content":"机器学习是AI的核心","tags":["AI","机器学习"]},
    )
    store.put(
        ("notes","datas"),
        "note_3",
        {"title":"Git使用技巧","content":"Git是版本控制工具","tags":["git","编程"]},
    )
    @tool
    def search_notes(keyword:str)->str:
        """
        搜索笔记
        参数：
            keyword:搜索关键词
        """
        #使用filter过滤
        """
        filter:dict,如filter={"tags":["python"]}
        limit：int  最多返回多少条，默认为10
        offset:int,跳过N条结果，用于翻页
        query：str，自然语言查询，如query="机器学习应用"
        """
        #notes = store.search(("notes","datas"),filter={"tags":["python"]})
        notes = store.search(("notes","datas")) #搜索namespace下的所有条目
        results = []
        for note in notes:
            if (
                keyword.lower() in note.value.get("title","").lower()
                or keyword.lower() in note.value.get("content","").lower()
                or keyword.lower() in " ".join(note.value.get("tags",[])).lower()
            ):
                results.append(note.value)
        if not results:
            return f"未找到包含{keyword}的笔记"
        output = f"找到{len(results)}条笔记:\n"
        for i,note in enumerate(results,1):
            output += f"\n{i}.{note['title']}\n {note['content']}\n 标签:{','.join(note['tags'])}"
        return output
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model,
        tools=[search_notes],
        system_prompt="你是一个笔记管理株洲，帮助搜索和管理笔记",
        store=store,
    )
    print("\n=======测试1:搜索python相关的=====")
    result1 = agent.invoke(
        {"messages":[("user","搜索关于python的笔记")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n====测试2:搜索编程相关的======")
    result2 = agent.invoke(
        {"messages":[("user","搜索所有跟编程相关的内容")]},
        config={"callbacks":[langfuse_handler]},
    )
    print(f"AI:{result2['messages'][-1].content}")


def example_4():
    """
    目标：理解两种存储模式的区别
    - profile：单用户单一档案
    - collection 多条目集合
    """
    print("\n============示例4:profile vs collection 模式=============")
    store=get_postgres_store()
    @dataclass
    class UserContext:
        user_id: str
    @tool
    def set_user_preference(key:str,value:str,runtime:ToolRuntime[UserContext])->str:
        """
        设置用户偏好profile模式
        每个用户只有一个配置
        """
        user_id = runtime.context.user_id
        data = store.get(("prefs",),user_id)
        prefs = data.value if data else {}
        prefs[key] = value
        store.put(("prefs",),user_id, prefs)
        return f"已保存：{key} = {value}"
    @tool
    def add_bookmark(url:str,title: str,runtime:ToolRuntime[UserContext])->str:
        """
        添加书签collection模式
        用户可以有多个书签
        """
        user_id = runtime.context.user_id
        bookmarks = store.search(("bookmarks",user_id))
        bookmark_id = len(bookmarks) + 1
        store.put(("bookmarks",user_id),f"bm_{bookmark_id}",{"url":url,"title":title})
        return f"已添加书签：{title} {url}"
    @tool
    def list_bookmarks(runtime:ToolRuntime[UserContext])->str:
        """列出所有标签"""
        user_id= runtime.context.user_id
        bookmarks = store.search(("bookmarks",user_id))
        if not bookmarks:
            return "暂无标签"
        output = "书签列表:\n"
        for bm in bookmarks:
            output += f"-{bm.value['title']}:{bm.value['url']}\n"
        return output
    model = init_chat_model(
        base_url=os.getenv('ARK_BASE_URL'),
        api_key=os.getenv('ARK_API_KEY'),
        model_provider="openai",
        model="Doubao-Seed-2.0-lite",
        temperature=0
    )
    agent = create_agent(
        model=model,
        tools=[set_user_preference,add_bookmark,list_bookmarks],
        system_prompt="你是一个个人助手，管理用户的偏好和书签",
        context_schema=UserContext,
        store=store,
    )
    print("\n====测试1:profile模式==========")
    result1 = agent.invoke(
        {"messages":[("user","把我的主题设置为暗色")]},
        config={"configurable":{"thread_id":"profile_demo"},"callbacks":[langfuse_handler]},
        context=UserContext(user_id="U001"),
    )
    print(f"AI:{result1['messages'][-1].content}")
    print("\n======测试2:collection模式================")
    result2 = agent.invoke(
        {"messages":[("user","添加一个书签:github,网站：github.com")]},
        config = {"configurable":{"thread_id":"profile_demo"},"callbacks":[langfuse_handler]},
        context = UserContext(user_id="U001"),
    )
    print(f"AI:{result2['messages'][-1].content}")





def main(example_number:int):
    print("="*60)
    print("第9课-长期记忆")
    print("="*60)
    example={
        1:example_1,
        2:example_2,
        3:example_3,
        4:example_4,
        #5:example_5,
        #6:example_6
    }
    if example_number in example:
        example[example_number]()
    else:
        print(f"错误：实例编号{example_number}不存在")
if __name__ == "__main__":
    main(4)