
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

## dynamic 执行流程
这个模块是 NVIDIA RecSys Examples 中最重要的子系统之一，用于在 GPU 上实现动态 embedding 表管理（支持缓存、淘汰策略、分布式训练等）
下面给你画一张 真正基于源码结构的完整执行路径图（从 Python → C++ → CUDA → GPU 内存）

                       ┌──────────────────────────────┐
                       │  Training / Inference Model  │
                       │ examples/hstu/modules        │
                       │ embedding.py                 │
                       └───────────────┬──────────────┘
                                       │
                                       │
                       ┌───────────────▼────────────────┐
                       │ PyTorch Interface Layer        │
                       │                                │
                       │ BatchedDynamicEmbeddingTablesV2│
                       │ dynamicemb/batched_dynamic_... │
                       └───────────────┬────────────────┘
                                       │
                                       │
                       ┌───────────────▼────────────────┐
                       │ Config Layer                   │
                       │ DynamicEmbTableOptions        │
                       │ dynamicemb_config.py          │
                       │                                │
                       │  - cache size                  │
                       │  - eviction strategy           │
                       │  - initializer                 │
                       └───────────────┬────────────────┘
                                       │
                                       │
                       ┌───────────────▼────────────────┐
                       │ Table Management Layer         │
                       │                                │
                       │ KeyValueTable                  │
                       │ key_value_table.py             │
                       │                                │
                       │  - insert(key,value)           │
                       │  - lookup(key)                 │
                       │  - delete(key)                 │
                       └───────────────┬────────────────┘
                                       │
                                       │
                       ┌───────────────▼────────────────┐
                       │ Hash Table Backend             │
                       │                                │
                       │ HKVVariable                    │
                       │ src/hkv_variable.h             │
                       │                                │
                       │ GPU/CPU hierarchical hash map  │
                       └───────────────┬────────────────┘
                                       │
                                       │
                       ┌───────────────▼────────────────┐
                       │ Dynamic Variable Base          │
                       │                                │
                       │ dynamic_variable_base.h        │
                       │                                │
                       │ embedding storage manager      │
                       └───────────────┬────────────────┘
                                       │
                                       │
                ┌──────────────────────▼─────────────────────┐
                │ Storage Layer                               │
                │                                             │
                │ GPU HBM cache (hot embeddings)              │
                │ Host DRAM storage (cold embeddings)         │
                │                                             │
                │ eviction policy: LRU / LFU                  │
                └──────────────────────┬─────────────────────┘
                                       │
                                       │
                ┌──────────────────────▼─────────────────────┐
                │ CUDA Kernel Layer                           │
                │                                             │
                │ embedding_lookup_kernel                     │
                │ hash_lookup_kernel                          │
                │ embedding_pooling_kernel                    │
                │                                             │
                │ parallel lookup + fused ops                 │
                └──────────────────────┬─────────────────────┘
                                       │
                                       │
                           ┌───────────▼───────────┐
                           │ GPU Memory (HBM)      │
                           │ embedding cache       │
                           │ optimizer states      │
                           └───────────────────────┘


## 理解EmbeddingCollection功能和jagged tensor 


### 一、准备：定义 embedding 表

我们有 3 个特征：

| feature | table      | vocab大小 | dim |
| ------- | ---------- | ------- | --- |
| user_id | user_table | 1000    | 2   |
| item_id | item_table | 1000    | 3   |
| tag（多值） | tag_table  | 100     | 2   |

---

#### 假设 embedding 权重（手写出来）

##### user_table（dim=2）

```text
id : embedding
1  → [0.1, 0.2]
2  → [0.3, 0.4]
```

---

##### item_table（dim=3）

```text
10 → [1.0, 1.1, 1.2]
20 → [2.0, 2.1, 2.2]
30 → [3.0, 3.1, 3.2]
```

---

##### tag_table（dim=2）

```text
5 → [0.5, 0.5]
6 → [0.6, 0.6]
7 → [0.7, 0.7]
```

---

### 二、输入数据（KeyedJaggedTensor 逻辑结构）

假设 batch_size = 2：

```text
样本1:
  user_id = 1
  item_id = 10
  tag = [5, 6]

样本2:
  user_id = 2
  item_id = 20
  tag = [7]
```

---

#### 转成 KJT 的逻辑表示：

```python
{
  "user_id": [1, 2]
  "item_id": [10, 20]
  "tag": [
    [5, 6],   # sample1
    [7]       # sample2
  ]
}
```

👉 tag 是 **变长（Jagged）**

---

### 三、Step 1：EmbeddingCollection 查表（lookup）

#### 1️⃣ user_id lookup

```text
[1, 2]
↓
[[0.1, 0.2],
 [0.3, 0.4]]
```

shape:

```text
[2, 2]
```

---

#### 2️⃣ item_id lookup

```text
[10, 20]
↓
[[1.0, 1.1, 1.2],
 [2.0, 2.1, 2.2]]
```

shape:

```text
[2, 3]
```

---

#### 3️⃣ tag lookup（注意：还没 pooling）

```text
sample1: [5,6] → [[0.5,0.5], [0.6,0.6]]
sample2: [7]   → [[0.7,0.7]]
```

---

### 四、Step 2：Pooling（关键）

EmbeddingCollection 默认会对 jagged 特征做 pooling（如 sum）

---

#### tag pooling（sum）

##### sample1

```text
[0.5,0.5] + [0.6,0.6]
= [1.1, 1.1]
```

---

##### sample2

```text
[0.7,0.7]
```

---

#### pooling 后结果：

```text
[[1.1, 1.1],
 [0.7, 0.7]]
```

shape:

```text
[2, 2]
```

---

### 五、Step 3：输出（KeyedTensor）

EmbeddingCollection 输出：

```python
{
  "user_id": tensor([
    [0.1, 0.2],
    [0.3, 0.4]
  ]),

  "item_id": tensor([
    [1.0, 1.1, 1.2],
    [2.0, 2.1, 2.2]
  ]),

  "tag": tensor([
    [1.1, 1.1],
    [0.7, 0.7]
  ])
}
```

---

### 六、Step 4：如果进入推荐模型（关键连接）

通常会做：

```python
# 拼接
x = concat([
  user_emb,   # [2,2]
  item_emb,   # [2,3]
  tag_emb     # [2,2]
])
```

得到：

```text
shape = [2, 7]
```

## 理解：recSdk中的BatchedDynamicEmbeddingTablesV2

gitcode上的代码：https://gitcode.com/Ascend/RecSDK/blob/develop/training/torch_rec_v2/dynamic_emb/dynamic_emb/distributed/batched_dynamicemb_table.py


### 总结

多表、多 bag、多 id 的动态 embedding 查表执行器

#### 处理流程
* 输入：
**(indices, offsets)**
* 处理：
ids -> internal index
不存在则动态创建
批量查 embedding
可选 pooling / unique 优化
* 输出：
每张表 / 每个样本 对应的 embedding 表示

#### 模拟输入数据示例

* batch数据，两个样本：
  sample0:
    user_id    = [1001]
    item_id    = [11, 12]
    keyword_id = [5, 6, 7]

  sample1:
    user_id    = [1002]
    item_id    = [13]
    keyword_id = [8]

* 转成indices和offsets
  user_id:
  indices = [1001, 1002]
  offsets = [0, 1, 2]

  item_id:
  indices = [11, 12, 13]
  offsets = [0, 2, 3]

  tag_id:
  indices = [5, 6, 7, 8]
  offsets = [0, 3, 4]

* 模拟送入模型的代码：

```python

model = BatchedDynamicEmbeddingTablesV2({
    "user_id": 4,
    "item_id": 4,
})

inputs = {
    "user_id": {
        "indices": user_indices,
        "offsets": user_offsets,
    },
    "item_id": {
        "indices": item_indices,
        "offsets": item_offsets,
    }
}

outputs = model(inputs)

print("user_id output shape:", outputs["user_id"].shape)
print(outputs["user_id"])
print("item_id output shape:", outputs["item_id"].shape)
print(outputs["item_id"])

  ```

  
  







