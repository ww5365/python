
# 

## feature_id space

推荐平台的feature_id space 的理解？
咨询chatgpt：https://chatgpt.com/c/69b7be9e-c338-8323-aefd-6e4f531021ed

feature_id space 不是特征个数，是特征取值的总数。

比如：我有两个特征： 爱好和性别， 爱好取值：20个  性别取值：2 个  那么我的feature_id space是20+2=22.
也就是我的token 总数是22个。 

| 术语               | 含义       |
|------------------|----------|
| feature          | 字段       |
| feature value    | token    |
| feature_id       | token 编号 |
| feature_id space | token总数  |


这张表这么大，我该怎么处理？高效的支持推荐系统模型的训推。




##  dynamic embedding

推荐系统，工作界面临的大难题：embedding的规模   feature_id space巨大，假设为10亿。单张embedding table大小：
100亿*128维*4byte 约为 5TB  GPU的显存根本装不下。


传统解决方法：

- parameter server   
- cpu embedding   : 问题？gpu访问cpu embedding 通过PCIe 延迟非常高
- ssd embedding

出现了dynamic embedding(动态embedding缓存) 方案，核心思想：
* 只把热点 embedding 放到gpu   : L1 cache
* 冷数据放cpu/ssd             : L2 storage



## MindSpeed Megatron-LM  torchrec  torch-npu  大模型组件

[deepseek参考](https://chat.deepseek.com/a/chat/s/e3059fd5-a995-43f7-ae26-0aa1efe94318)

| 组件名称 | 核心定位 | 一句话描述 |
| :--- | :--- | :--- |
| **MindSpeed** | 华为昇腾NPU的大模型训练加速库 | 专为昇腾打造，让Megatron-LM等框架能在NPU上高效训练大模型。 |
| **Megatron-LM** | NVIDIA GPU的大模型训练框架 | 由NVIDIA推出，定义了大规模分布式训练的标准，是行业标杆。 |
| **TorchRec** | PyTorch的推荐系统专用库 | 专注于解决推荐系统中大规模嵌入表（稀疏特征）的训练痛点。 |
| **torch-npu** | 华为昇腾NPU的PyTorch插件 | 作为一个桥梁，让PyTorch能够识别和调用昇腾NPU进行计算。 |



