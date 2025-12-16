

# 




Agent Lightning 的架构设计保持了组件的最小化。你的 Agent 照常运行，通过一个轻量级的 agl.emit_xxx() 辅助函数或追踪器来收集提示、工具调用和奖励等事件。
这些事件被构造成结构化的 spans，流入 LightningStore（一个中心枢纽）。 
算法模块从 LightningStore 读取 spans 进行学习，并将**优化后的资源（如精炼的提示词模板或新的策略权重）发布**回去。


Agent lightning最终优化的是什么？
1. 提示词模版   这个能理解
2. 新的策略权重，这个怎么理解？




https://zhuanlan.zhihu.com/p/1937109083623782314





