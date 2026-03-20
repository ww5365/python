
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


## hstu模型训练

### 训练数据

input_items   = [1193, 661, 914]
input_actions = [1,    0,   1]



#### embedding table
1️⃣ item embedding

```text
E_item(1193) = [0.2, 0.1, 0.4, 0.7]
E_item(661)  = [0.5, 0.3, 0.2, 0.1]
E_item(914)  = [0.6, 0.9, 0.1, 0.4]
```

---

2️⃣ action embedding

（只有2种：喜欢 / 不喜欢）

```text
E_action(1) = [0.3, 0.3, 0.3, 0.3]   # positive
E_action(0) = [0.1, 0.1, 0.1, 0.1]   # negative
```

---

3️⃣ position embedding

```text
E_pos(0) = [0.01, 0.02, 0.03, 0.04]
E_pos(1) = [0.05, 0.06, 0.07, 0.08]
E_pos(2) = [0.09, 0.10, 0.11, 0.12]
```


#### 构造token

🔹 位置 t1

```text
token_1 =
[0.2, 0.1, 0.4, 0.7]   (item)
+ [0.3, 0.3, 0.3, 0.3] (action)
+ [0.01,0.02,0.03,0.04](pos)
```

```text
= [0.51, 0.42, 0.73, 1.04]
```

其它位置类似

#### 喂给hstu模型的数据

```text
sequence = [
  [0.51, 0.42, 0.73, 1.04],
  [0.65, 0.46, 0.37, 0.28],
  [0.99, 1.30, 0.51, 0.82]
]
```


