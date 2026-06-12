
## 问题1：训练脚本中全局变量:

根据脚本代码，用户定义的全局变量（通过 `tf.get_variable` 或 `tf.Variable` `tf.layers.dense`创建）主要分为三类，均以 **`float32`** 存储，说明如下。

### 一、合并 Embedding 表变量
每个合表（`MergedTable`）创建一个 embedding 参数变量，变量名模式为 `merged_embedding_table_<table_id>`。  
- **形状**：`[global_capacity, embedding_dim]`，其中 `global_capacity` 为该合表包含的所有小表容量之和，`embedding_dim` 是配置的嵌入维度。  
- **数据类型**：`tf.float32`  
- **个数**：等于合并后的大表数量（由 `groups` 决定）。

### 二、DCN 网络参数变量
`dcn_model()` 中显式创建的权重与偏置，以及输出层。

| 变量名 | 形状 | 数据类型 | 说明 |
|--------|------|---------|------|
| `cross_w_0` ~ `cross_w_2` | `[input_dim, 1]` | `tf.float32` | Cross 层的权重（共 3 层） |
| `cross_b_0` ~ `cross_b_2` | `[input_dim]` | `tf.float32` | Cross 层的偏置 |
| `deep_w_0` ~ `deep_w_4` | 见注释 | `tf.float32` | Deep 层的权重（共 5 层，逐层维度变化） |
| `deep_b_0` ~ `deep_b_4` | 对应层输出维度 | `tf.float32` | Deep 层的偏置 |
| `dense/kernel` | `[concat_dim, 1]` | `tf.float32` | 最终线性输出层的权重 |
| `dense/bias` | `[1]` | `tf.float32` | 最终线性输出层的偏置 |

> **注**：`input_dim = len(total_slot) * embedding_dim`；Deep 层的维度依次为 `[input_dim, 2950]`、`[2950, 2048]`、`[2048, 1024]`、`[1024, 512]`、`[512, 256]`；`concat_dim = input_dim + 256`。


### 三、优化器状态（Slot）变量
两个优化器（稀疏优化器 `CustomizedLazyAdam` 和密集优化器 `AdamOptimizer`）为各自管理的参数创建的一阶 / 二阶矩估计变量，通常命名包含 `m`、`v` 或类似后缀。  
- **稀疏优化器 Slot**：为每个 `merged_embedding_table_*` 创建两份状态（`m` 和 `v`），形状与对应 embedding 表相同。  
- **密集优化器 Slot**：为 DCN 的所有可训练参数各创建两份状态，形状与对应变量相同。  
- **数据类型**：均为 `tf.float32`。  
- **注意**：这些变量虽非直接调用 `tf.get_variable` 创建，但由优化器隐式生成并加入全局变量集合，是实际显存的重要组成部分。

### 总结
- **变量总数**取决于合表数量、网络深度和优化器，全部基于 `float32`。  
- 日志中 `tf_var_est` 的 `emb`、`dense`、`slot` 三项即分别统计以上三类变量（因脚本仅统计了名称包含特定关键字的全局变量，故分类与上文一致）。