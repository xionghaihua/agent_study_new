## LangChain框架  

### 开源第三方库
- langchain-core ：基础抽象和LangChain表达式语言
- langchain-community ：第三方集成。合作伙伴包（如langchain-openai、langchain-anthropic等），一些集成已经进一步拆分为自己的轻量级包，只依赖于langchain-core
- langchain ：构成应用程序认知架构的链、代理和检索策略
- langgraph：通过将步骤建模为图中的边和节点，使用 LLMs 构建健壮且有状态的多参与者应用程序
- langserve：将 LangChain 链部署为 REST API
- LangSmith：一个开发者平台，可让您调试、测试、评估和监控LLM应用程序，并与LangChain无缝集成

langchain-core核心功能
- runnable
- 消息系统
- 提示词模版
- 输出解析器
- 缓存接口

langchain-community: 社区工具
langgraph


-------
pip install langchain==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-openai==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-community -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install dashscope



