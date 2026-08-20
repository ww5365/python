import os
import time
import numpy as np
import tensorflow as tf
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
from tensorflow.python.framework import graph_util
import horovod.tensorflow as hvd
from tensorflow.python.framework import ops
import tensorflow.python.ops.math_ops as math_ops 
import math

from sparse_optimizer import *


import sys
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()

from logger_utils import setup_logger


device_ids = os.environ.get('NPU_IDS', '0,1,2,3,4,5,6,7').split(" ")  # 确保2个设备

rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
os.environ['ASCEND_DEVICE_ID'] = device_ids[rank]
os.environ['WORKER_DEVICE_ID'] = device_ids[rank]

# ---- 初始化 logger（每个 rank 单独文件） ----
logger = setup_logger(rank)
logger.info(f"start rank: {rank}")

import random
seed = 1234
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


#general_slot =  [1, 68, 1, 1, 1, 72, 1, 1, 1, 1000, 1, 59, 1, 81, 1, 1, 47, 1, 1, 93, 1, 1, 77, 1, 1, 62, 1, 1, 88, 1, 53, 1, 1, 98, 1, 1, 71, 1, 84, 1, 1, 65, 1, 1, 91, 1, 1, 80, 1, 1] # 2222

general_slot =  [3, 4]

bs = int(os.environ.get('bacth_size', 10))
# all2all_slot = [50] * 50
all2all_slot = general_slot # [50] * 50
total_slot = all2all_slot
total_all2all_slot_size = sum(all2all_slot)  # 5
embedding_dim = 8
Iters = 15

# uplicate_rate = 0.9
# v_size = int(bs*(1-uplicate_rate)*50)
# local_vocabulary_size = [v_size] * len(all2all_slot)   # 重复度90%，bs*slot[i]


local_vocabulary_size = [50] * len(all2all_slot)

path = os.path.join(os.getcwd(), 'checkpoint')

num_npus =  8              # NPU卡数量
uplicate_rate = 0.6
# conbiner = 0  # 求和
conbiner = 1  # 平均


prefetch_step = 0
is_padding = False
padding_key = -1
is_completion = False
completion_key = 99
default_keys =  [-3] * len(total_slot)

# ADDED LOGGER: 记录关键超参
logger.info(f"Hyperparameters: bs={bs}, embedding_dim={embedding_dim}, Iters={Iters}, "
            f"local_vocabulary_size={local_vocabulary_size}, all2all_slot={all2all_slot}, "
            f"num_npus={num_npus}, uplicate_rate={uplicate_rate}, conbiner={conbiner}")

# 输入数据
local_indices = []
for i in range(len(all2all_slot)):
    start  = 0
    end =  local_vocabulary_size[i]  
    indices = np.random.randint(start, end, size=(bs, all2all_slot[i]),
                                dtype=np.int64)  # bs: 3   slot: [4, 5]
    
    local_indices.append(indices)    

# 生成随机二分类标签（用于AUC计算）
local_labels = np.random.randint(0, 2, size=(bs, 1)).astype(np.float32)

# ADDED LOGGER: 1. 记录输入数据形状和部分内容

logger.info("=============================================part1==============================================================")

for i, indices in enumerate(local_indices):
    logger.info(f"Slot {i} input indices shape and content======\n {indices.shape}\n dtype: {indices.dtype}\n row:\n {indices}")

logger.info(f"local_labels shape: {local_labels.shape}, dtype: {local_labels.dtype}")
logger.info(f"local_labels content : {local_labels}")
    

def create_all2all_embedding_for_every_slot(sfps):
    """
        Note: create_embedding API should not be used mixedly with native api, like remote_embedding, xxx_embedding...
    """

    logger.info("=============================================part2==============================================================")
    
    for i in range(len(all2all_slot)):
        init = sfps.Initializer(10, 10, 'constant')
        lrs = sfps.ConstantLR(lr=0.005)
        opt = sfps.Adam(lrs, decay=0.0)
        comm_policy = sfps.communication_policy(communication_type='allreduce', grad_aggregation_type='average')

        feature_policy = sfps.feature_policy('counter_filter_with_default',
                                             counter_threshold=0,
                                             default_key=default_keys[i], shrink_type='step',
                                             shrink_step_threshold=0)

        padding = sfps.padding_param(padding=is_padding, key=padding_key, mask=False)
        feature_completion_param = sfps.feature_completion_param(
            completion=is_completion,
            key=completion_key
        )
        
        # ---- 新增 logger 记录表信息 ----
        logger.info(
            f"Creating all2all table[{i}]: "
            f"vocab_size={local_vocabulary_size[i]}, emb_dim={embedding_dim}, "
            f"slot_len={all2all_slot[i]}, batch_size={bs}, "
            f"init=constant(10,10), opt=Adam(lr=0.005, decay=0.0), "
            f"comm=allreduce(average), feature_filter=counter_filter, "
            f"default_key={default_keys[i]}, padding={is_padding}(key={padding_key}), "
            f"completion={is_completion}(key={completion_key})"
        )

        sfps.create_table(sfps.c_lib.embedding_type.all2all, local_vocabulary_size[i], embedding_dim,
                          bs, [all2all_slot[i]], opt, init, sfps.c_lib.key_type.int64, sfps.c_lib.pooling_type.sum,
                          sfps.c_lib.hash_type.hash, comm_policy, feature_policy, None)


def create_embeddings(vocab_sizes, embedding_dim, slot_vec, conbiner, num_npus, base_table_id, sparse_optimizer,
                      pooling=True):
    # 创建所有embedding tables
    embedding_engines = []
    embedding_tables = []

    logger.info(
        "=============================================part3==============================================================")

    for i, vocab_size in enumerate(vocab_sizes):
        embEngine = EmbeddingEngine(
            local_vocab_size=local_vocabulary_size[i],
            embedding_dim=embedding_dim,
            slot_vec=[slot_vec[i]],
            conbiner=conbiner,
            num_npus=num_npus,
            rank=rank,
            table_id=base_table_id + i,
            sparse_optimizer=sparse_optimizer,
            pooling=pooling
        )

        embedding_tables.append(embEngine.embedding_table)
        embedding_engines.append(embEngine)

        # 从 tf.Variable 中提取名称、形状、数据类型
        var_name = embEngine.embedding_table.name
        var_shape = embEngine.embedding_table.shape
        var_dtype = embEngine.embedding_table.dtype

        logger.info(
            f"Created EmbeddingEngine[{i}]: "
            f"table_id={base_table_id + i}, "
            f"local_vocab_size={local_vocabulary_size[i]}, "
            f"embedding_dim={embedding_dim}, "
            f"slot_vec={slot_vec[i]}, "
            f"conbiner={conbiner}, "
            f"pooling={pooling}, "
            f"embedding_table name={var_name}, "
            f"shape={var_shape}, "
            f"dtype={var_dtype}"
        )

    return embedding_tables, embedding_engines


def get_input(worker_rank, step, uplicate_rate=0.6):
    return local_indices, local_labels   # 同时返回标签


def create_hash_optimizer(learning_rate=0.1):
    return CustomizedLazyAdam(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        name="LazyAdam"
    )

def model(batch_embedding, label):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        hidden_size = [len(total_slot) * (embedding_dim ), 512, 128, 1]
        output = tf.reshape(batch_embedding, [-1, len(total_slot) * (embedding_dim )])
        for i in range(len(hidden_size) - 1):
            w = tf.get_variable(
                'w'+str(i),
                shape=[hidden_size[i], hidden_size[i + 1]],
                #initializer=tf.zeros_initializer(),
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                # initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01, seed=1234),
                #initializer = tf.constant_initializer(value=10),
                dtype=tf.float32,
                trainable=True
            )
            
            output = tf.nn.relu(tf.matmul(output, w))
        loss = tf.reduce_mean(tf.square(output - label), name='loss')
    
    return loss


def dcn_model(batch_embedding, label, is_pooling=True):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        if is_pooling:
            pooled_tensor = batch_embedding
        else:
            # dps用非池化场景，池化操作在dense层完成
            split_size = tf.constant(total_slot,dtype=tf.int32)
            splitted = tf.split(batch_embedding, split_size, axis=1)
            pooled = [tf.reduce_mean(embedding, axis=1) for embedding in splitted]
            pooled_tensor = tf.stack(pooled, axis=1)
        
        input_dim = len(total_slot) * embedding_dim
        output = tf.reshape(pooled_tensor, [-1, input_dim])

        # ===== 1. Cross Network (显式特征交叉) =====
        cross_depth = 3  # 交叉层数
        x0 = xl = output
        for i in range(cross_depth):
            w = tf.get_variable(f'cross_w_{i}', shape=[input_dim, 1],
                                initializer=tf.glorot_normal_initializer())
            b = tf.get_variable(f'cross_b_{i}', shape=[input_dim],
                                initializer=tf.zeros_initializer())
            xl = x0 * tf.matmul(xl, w) + xl + b  # 交叉公式: x_{l+1} = x0 * (xl^T w) + xl + b

        # ===== 2. Deep Network (隐式高阶交叉) =====
        deep_dims = [512, 128]  # 深度网络层维度
        # deep_dims = [128, 32, 8, 4]  # 深度网络层维度
        # deep_dims = [32]  # 深度网络层维度
        x_deep = output
        for i, dim in enumerate(deep_dims):
            w = tf.get_variable(f'deep_w_{i}', shape=[x_deep.shape[-1], dim],
                                initializer=tf.glorot_normal_initializer())
            b = tf.get_variable(f'deep_b_{i}', shape=[dim],
                                initializer=tf.zeros_initializer())
            x_deep = tf.nn.relu(tf.matmul(x_deep, w) + b)

        # ===== 3. 合并两部分输出 =====
        concat = tf.concat([xl, x_deep], axis=1)
        logits = tf.layers.dense(concat, 1, activation=None)

        # ===== 4. 定义输出节点 =====
        prediction = tf.identity(logits, name='prediction_output')  # 添加输出节点

        loss = tf.reduce_mean(tf.square(logits - label), name='loss')

    # 修改：返回 loss 和 logits（用于计算AUC）
    return loss, logits


def allreduce(grads, average=True, compression=None):
    if get_rank_size() == 1:
        return grads
    averaged_gradients = []
    with tf.name_scope("Allreduce"):
        for grad, var in grads:
            if grad is not None:
                avg_grad = hccl_ops.allreduce(grad, "sum")
                averaged_gradients.append((avg_grad, var))
            else:
                averaged_gradients.append((None, var))
    return averaged_gradients


# --------------------------
# 1. HCCL通信工具函数（使用hccl_ops.all_to_all_v）
# --------------------------
def alltoallv_exchange_sizes(send_sizes, num_npus, dtype=tf.int32):
    """通过hccl_ops.all_to_all_v交换各卡发送数据量，获取接收大小"""
    # 转换send_sizes为张量，确保格式正确
    send_sizes = tf.convert_to_tensor(send_sizes, dtype=dtype)
    
    # 发送配置：每个目标卡接收1个size值（send_sizes中的一个元素）
    send_counts = tf.ones([num_npus], dtype=tf.int64)  # 每个rank发送1个元素
    send_displacements = tf.range(num_npus, dtype=tf.int64)  # 发送偏移量：0,1,...,num_npus-1
    
    # 接收配置：从每个源卡接收1个size值
    recv_counts = tf.ones([num_npus], dtype=tf.int64)  # 每个rank接收1个元素
    recv_displacements = tf.range(num_npus, dtype=tf.int64)  # 接收偏移量：0,1,...,num_npus-1 [0,1)
    recv_buf = tf.zeros([num_npus], dtype=dtype)  # 接收缓冲区
    
    # 执行HCCL all_to_all_v交换size信息
    recv_buf = hccl_ops.all_to_all_v(
        send_data=send_sizes,
        send_counts=send_counts,
        send_displacements=send_displacements,
        recv_counts=recv_counts,
        recv_displacements=recv_displacements
    )
    
    return recv_buf


@tf.custom_gradient
def alltoallv_exchange_data(send_tensors, send_sizes, recv_sizes):
    """通过hccl_ops.all_to_all_v交换实际数据（支持动态大小）"""
    # 拼接发送张量为连续缓冲区（hccl_ops.all_to_all_v要求输入为单个张量）
    send_buf = tf.concat(send_tensors, axis=0)
    
    recv_sizes = tf.ensure_shape(recv_sizes, (None,))  # 静态约束1维
    
    # 发送配置：计算发送计数和偏移量（前缀和）
    send_counts = tf.cast(send_sizes, tf.int64)
    send_displacements = tf.cumsum(tf.concat([[0], tf.cast(send_sizes[:-1], tf.int64)], axis=0))
    
    # 接收配置：计算接收缓冲区大小和偏移量
    recv_total = tf.reduce_sum(recv_sizes)
    recv_buf = tf.zeros([recv_total], dtype=tf.int32)  # 预分配接收缓冲区
    recv_counts = tf.cast(recv_sizes, tf.int64)
    recv_displacements = tf.cumsum(tf.concat([[0], tf.cast(recv_sizes[:-1], tf.int64)], axis=0))
    
    # 执行HCCL all_to_all_v数据交换
    recv_buf = hccl_ops.all_to_all_v(
        send_data=send_buf,
        send_counts=send_counts,
        send_displacements=send_displacements,
        recv_counts=recv_counts,
        recv_displacements=recv_displacements
    )
    
    # 梯度函数：all-to-all的逆操作
    def grad(upstream_grad):
        grad_send = hccl_ops.all_to_all_v(
            send_data=upstream_grad,
            send_counts=recv_counts,  
            send_displacements=recv_displacements,
            recv_counts=send_counts,
            recv_displacements=send_displacements
        )
        
        
        return tf.split(grad_send, send_sizes, axis=0), None, None
    
    return  recv_buf, grad                            # tf.split(recv_buf, recv_sizes, axis=0)



@tf.custom_gradient
def alltoallv_exchange_embeddings(send_tensors, send_sizes, recv_sizes):
    """通过hccl_ops.all_to_all_v交换实际数据（支持动态大小）"""
 
    recv_sizes = tf.ensure_shape(recv_sizes, (None,))
    
    # 发送配置：计算发送计数和偏移量（前缀和）
    send_counts = tf.cast(send_sizes, tf.int64)
    send_displacements = tf.cumsum(tf.concat([[0], tf.cast(send_sizes[:-1], tf.int64)], axis=0))
    
    # 接收配置：计算接收缓冲区大小和偏移量
    recv_total = tf.reduce_sum(recv_sizes)
    recv_buf = tf.zeros([recv_total], dtype=tf.float32)  # 预分配接收缓冲区
    recv_counts = tf.cast(recv_sizes, tf.int64)
    recv_displacements = tf.cumsum(tf.concat([[0], tf.cast(recv_sizes[:-1], tf.int64)], axis=0))
    
    # 执行HCCL all_to_all_v数据交换
    recv_buf = hccl_ops.all_to_all_v(
        send_data=send_tensors,
        send_counts=send_counts,
        send_displacements=send_displacements,
        recv_counts=recv_counts,
        recv_displacements=recv_displacements
    )
    
    # 梯度函数：all-to-all的逆操作
    def grad(upstream_grad):
        grad_send = hccl_ops.all_to_all_v(
            send_data=upstream_grad,
            send_counts=recv_counts,
            send_displacements=recv_displacements,
            recv_counts=send_counts,
            recv_displacements=send_displacements
        )
        return grad_send, None, None
    
    return  recv_buf, grad                          



def efficient_unique(tensor, return_inverse=True):
    target_dtype = tf.int64
    
    if tf.size(tensor) == 0:
        if return_inverse:
            return tf.cast(tensor, target_dtype), tf.constant([], dtype=target_dtype)
        return tf.cast(tensor, target_dtype)
    
    sorted_tensor = tf.sort(tensor)
    sort_indices = tf.cast(tf.argsort(tensor), dtype=target_dtype)
    
    is_different = tf.not_equal(sorted_tensor[1:], sorted_tensor[:-1])
    is_unique = tf.concat([tf.constant([True]), is_different], axis=0)
    
    unique_tensor = tf.cast(tf.boolean_mask(sorted_tensor, is_unique), target_dtype)
    
    result = [unique_tensor]
    
    if return_inverse:
        unique_indices = tf.cumsum(tf.cast(is_unique, target_dtype), axis=0) - 1
        inverse_indices = tf.scatter_nd(
            indices=tf.expand_dims(sort_indices, axis=1), 
            updates=unique_indices,                        
            shape=tf.cast(tf.shape(sort_indices), dtype=target_dtype)
        )
        result.append(inverse_indices)
    
    return result[0] if len(result) == 1 else tuple(result)

# --------------------------
# 2. 嵌入表引擎核心类
# --------------------------
class EmbeddingEngine:
    def __init__(self, local_vocab_size, embedding_dim, slot_vec, conbiner, num_npus, rank, table_id, sparse_optimizer, pooling=True):
        self.table_id = table_id
        self.local_vocab_size = local_vocab_size  # 本地嵌入表最大容量
        self.embedding_dim = embedding_dim        # 嵌入维度
        self.num_npus = num_npus                  # 总卡数
        self.rank = rank                          # 当前卡序号
        self.slot_vec = slot_vec
        self.conbiner = conbiner
        self.sparse_optimizer = sparse_optimizer
        self.pooling = pooling   # True: 返回聚合后的向量；False: 返回原始序列
        
        # 用于保存中间张量，便于运行时打印
        self.debug_tensors = {}
        
        # # 初始化独立嵌入表（tf.Variable）
        with tf.variable_scope(f"embedding_table_scope_{rank}_table_{self.table_id}"):
            self.embedding_table = tf.get_variable(
                f'embedding_table_{self.table_id}',
                shape=[self.local_vocab_size, embedding_dim],            
                initializer=tf.constant_initializer(value=rank), # 自定义常量值
                # initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=1234),                   
                dtype=tf.float32
            )

    def lookup_and_gather(self, input_ids):
        """完整查询流程（基于动态offset）"""
        # 步骤1：去重并准备发送ID 
        # 步骤2：All2All交换ID
        # 步骤3：本地查询（基于动态offset）    
        self.send_sizes, self.recv_sizes, self.indices, self.uindex, offset_count = \
            _key_process(input_ids, self.table_id, slot_size=self.slot_vec[0],
                         name_=f'keyprocess{self.table_id}', is_prefetch_=False,
                         npu_num_=self.num_npus, cap_=self.local_vocab_size)
            
        # ---------- 保存中间张量用于运行时打印 ----------
        self.debug_tensors['offset_count'] = offset_count
        self.debug_tensors['uindex0'] = self.uindex 
        
        self.uindex = self.uindex[:offset_count]
        
        
        
        all_recv_embeds = tf.nn.embedding_lookup(self.embedding_table, self.uindex)
        
        
        # 步骤4： embedding all2all 
        send_embedding_sizes = self.recv_sizes * self.embedding_dim
        recv_embedding_sizes =  self.send_sizes * self.embedding_dim 
        # alltoallv_exchange_sizes(send_embedding_sizes, self.num_npus)                                         
        
        recv_embeds_list = alltoallv_exchange_embeddings(
                    all_recv_embeds,
                    send_embedding_sizes,
                    recv_embedding_sizes
        )
        
        recv_embeds_list = tf.reshape(recv_embeds_list, [-1, self.embedding_dim])
        
        
        # # 步骤6 gather 
       
        restored_embeddings = tf.gather(recv_embeds_list, self.indices) # 可以取到30个维度为8的embedding值 [30, 8]
        final_restored = tf.reshape(restored_embeddings, [bs, -1, self.embedding_dim]) # [10,3,8]
        
        # ---------- 保存中间张量用于运行时打印 ----------
        self.debug_tensors['send_sizes'] = self.send_sizes
        self.debug_tensors['recv_sizes'] = self.recv_sizes
        self.debug_tensors['indices'] = self.indices
        self.debug_tensors['uindex'] = self.uindex
        self.debug_tensors['all_recv_embeds'] = all_recv_embeds
        self.debug_tensors['recv_embeds_list'] = recv_embeds_list
        self.debug_tensors['final_restored'] = final_restored

        if self.pooling:
            # 池化模式：按 slot 聚合（原有逻辑）
            cum_sizes = [0]
            current = 0
            for size in self.slot_vec:
                current += size
                cum_sizes.append(current)
            starts = cum_sizes[:-1] # 起始索引：[0,1,3]
            ends = cum_sizes[1:] # 结束索引：[1,3,6]

            slot_aggs = []
            for s, e in zip(starts, ends):
                slot = final_restored[:, s:e, :] # 形状：[bs, slot_len, dim]
                if self.conbiner == 0:
                    slot_aggs.append(tf.reduce_sum(slot, axis=1))
                else:
                    slot_aggs.append(tf.reduce_mean(slot, axis=1))
            result = tf.stack(slot_aggs, axis=1)   # [bs, num_slots, dim]
        else:
            # 非池化模式：直接返回序列
            result = final_restored                # [bs, total_slot_len, dim]
                
        return restored_embeddings, result
    
    
    def backward(self, embedding_grad, learning_rate=0.01):
        """
        反向梯度更新流程：
        1. 本地相同key梯度聚合（基于unique_embeddings_ids）
        2. 按分区策略进行梯度与ID的All2All跨卡交换
        3. 接收端按全局ID二次聚合（合并跨卡相同key的梯度）
        4. 映射为本地offset并更新嵌入表
        """
        # --------------------------
        # 1. 本地梯度聚合（相同key的梯度求和）
        # --------------------------
        # 重后的全局ID（与embedding_grad一一对应）
        # 对相同ID的梯度进行求和
        
        if isinstance(embedding_grad, tf.IndexedSlices):
            target_length = tf.shape(self.indices)[0]  
            target_shape = [target_length, self.embedding_dim]
            
            sparse_grad = tf.zeros(target_shape, dtype=embedding_grad.values.dtype)
            
            sparse_indices = tf.expand_dims(embedding_grad.indices, axis=1)
            sparse_grad = tf.tensor_scatter_nd_update(
                tensor=sparse_grad,
                indices=sparse_indices,
                updates=embedding_grad.values
            )
        else:
            sparse_grad = embedding_grad
        
        
        local_aggregated_grad = tf.math.unsorted_segment_sum(
            data=sparse_grad,
            segment_ids=self.indices,
            num_segments=tf.reduce_sum(self.send_sizes)
        )
        
        
        # # # --------------------------
        # # # 2. 梯度与ID的All2All跨卡交换
        # # # --------------------------
        # #  [xxx * embedding_dim]
        after_partition_grads = tf.reshape(local_aggregated_grad, [-1, self.embedding_dim])
        # # # 交换梯度数据
        receive_embedding_sizes = self.recv_sizes * self.embedding_dim
        send_embedding_sizes =  self.send_sizes * self.embedding_dim
        recv_all_grads = alltoallv_exchange_embeddings(
            after_partition_grads,
            send_embedding_sizes,
            receive_embedding_sizes
        )
        
        
        # # # --------------------------
        # # # 3. 接收端二次聚合（跨卡相同key的梯度求和）
        # # # --------------------------
        # # # 合并所有接收的全局ID和梯度
        global_grad_list = tf.reshape(recv_all_grads, [-1, self.embedding_dim])
        
        unique_uindex, idx_mapping = tf.unique(self.uindex)
        
        global_aggregated_grad = tf.math.unsorted_segment_sum(
			data=global_grad_list,
			segment_ids=idx_mapping,  
			num_segments=tf.shape(unique_uindex)[0]
		)
        
        
        final_sparse_grad = ops.IndexedSlices(
            values=global_aggregated_grad,
            indices=unique_uindex,
            dense_shape=tf.shape(self.embedding_table)
        )
        
        
        
        grad_var_pair = [(final_sparse_grad, self.embedding_table)]
        update_op = self.sparse_optimizer.apply_gradients(grad_var_pair)
        
        # ---------- 保存反向中间张量 ----------
        self.debug_tensors['embedding_grad'] = embedding_grad
        self.debug_tensors['sparse_grad'] = sparse_grad
        self.debug_tensors['local_aggregated_grad'] = local_aggregated_grad
        self.debug_tensors['after_partition_grads'] = after_partition_grads
        self.debug_tensors['recv_all_grads'] = recv_all_grads
        self.debug_tensors['global_grad_list'] = global_grad_list
        self.debug_tensors['global_aggregated_grad'] = global_aggregated_grad
        self.debug_tensors['unique_uindex'] = unique_uindex
        self.debug_tensors['idx_mapping'] = idx_mapping
        self.debug_tensors['final_sparse_grad'] = final_sparse_grad
        
        
        # with tf.control_dependencies([update_op]):  # 确保更新完成后再返回
        #     updated_table = tf.identity(self.embedding_table)  
        return update_op
        

if __name__ == '__main__':
    assert "worker" == os.getenv('DMLC_ROLE', "worker")
    from SFPS.tensorflow.TFWorker import TFWorker
    from SFPS.tensorflow.ops import _key_process
    
    # 全局开关：True 表示 pull 接口返回聚合特征，False 表示返回原始序列
    use_pooling_in_pull = False   # 改为 False 即启用非池化模式

    sfps = TFWorker()
    sfps.prefetch(prefetch_step)
    create_all2all_embedding_for_every_slot(sfps)
    npu_int = npu_ops.initialize_system()  # 初始化 NPU 集群环境
    npu_shutdown = npu_ops.shutdown_system() #  关闭 NPU 集群环境
    sfps.total_embedding_count = len(sfps.table_create_infos)
    #init_op = tf.global_variables_initializer()
    sfps.barrier()
    
    logger.info(f"sfps.total_embedding_count : {sfps.total_embedding_count}")   #这里长度是0
    
    sparse_optimizer = create_hash_optimizer(learning_rate=0.005)
    all2all_embedding_tables, embedding_engines = create_embeddings(
        local_vocabulary_size, embedding_dim, all2all_slot, conbiner,
        num_npus, sfps.total_embedding_count, sparse_optimizer,use_pooling_in_pull
    )

    # 标签占位符，替代固定 label
    label_placeholder = tf.placeholder(tf.float32, shape=[bs, 1], name='label_placeholder')
    local_input_placeholders = [
        tf.placeholder(tf.int64, shape=[bs, None], name=f'input_placeholder_{i}')
        for i in range(len(all2all_slot))
    ]
    
    
    all2all_embeddings = []
    all2all_embeddings_before_gather = []
    forward_emb_ops = []
    
    for i, placeholder in enumerate(local_input_placeholders):
        restored_embeddings,result  = embedding_engines[i].lookup_and_gather(placeholder)
        
        all2all_embeddings_before_gather.append(restored_embeddings)  # [30, 8] [40, 8]
        all2all_embeddings.append(result) # [10,3,8] [10,4,8]
        
        # forward_emb_ops.append(restored_embeddings.op) 
        # forward_emb_ops.append(result.op)
        
          
    # forward_emb_group_op = tf.group(forward_emb_ops, name="forward_emb_parallel")
    # with tf.control_dependencies([forward_emb_group_op]):
    all2all_embedding = tf.concat(all2all_embeddings, axis=1) # [10, 7, 8]
    batch_embedding = all2all_embedding
    logger.info(f"batch_embedding static shape: {batch_embedding.shape}")
    loss, logits = dcn_model(batch_embedding, label_placeholder, is_pooling=use_pooling_in_pull)

    # 预测概率，用于AUC计算
    pred_prob = tf.sigmoid(logits, name='pred_prob')

    # AUC 指标（累计）
    auc_value, auc_update_op = tf.metrics.auc(labels=label_placeholder, predictions=pred_prob)

    # ADDED LOGGER: 定义用于监控的张量
    batch_emb_mean = tf.reduce_mean(batch_embedding)
    batch_emb_std = tf.math.reduce_std(batch_embedding)
    batch_emb_shape = tf.shape(batch_embedding)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    var_list = tf.trainable_variables()
    # logger.info(f"-----var_list: {var_list}")
    dense_vars = var_list[len(all2all_slot):]   # [0, len(all2all_slot))是什么？embedding变量， 之后是dense层变量
    dense_grads = optimizer.compute_gradients(loss, dense_vars)
    sparse_grads = sparse_optimizer.compute_gradients(loss, all2all_embeddings_before_gather)
    
    # ADDED LOGGER: 记录可训练变量总数
    logger.info(f"Total trainable variables: {len(var_list)}")
    
    # # dense 层allreduce
    grads_and_vars = dense_grads
    grads_and_vars = allreduce(grads_and_vars)
    train_op = optimizer.apply_gradients(grads_and_vars)

    
    # # 按表更新梯度并更新embedding数据
    all2allGradOp = []
    all2allUpdateEmbedding = []
    sparse_grad_list = []
    with tf.control_dependencies([train_op]):
        for i in range(len(all2all_slot)):
            grad, var = sparse_grads[i] 
            updated_table = embedding_engines[i].backward(grad)
            all2allUpdateEmbedding.append(updated_table)
            
            # sparse_grad_list.append(sparse_grad)
        
    all2allUpdateEmbedding = tf.group(all2allUpdateEmbedding, name="train_step_group")
    
            
    sfps.train()
    
    #创建session，如果原始网络中使用了tf.device相关代码，则需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
    config = tf.ConfigProto()
    # config.gpu_options.visible_device_list = str(get_local_rank_id())
    config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.parameter_map["enable_parallel_graph"].b = True
    custom_op.parameter_map["parallel_graph_thread_pool_size"].i = 64
    
    custom_op.parameter_map["enable_parallel_fusion"].b = True  # 启用算子并行融合
    custom_op.parameter_map["enable_multi_stream"].b = True  # 开启多流支持
    custom_op.parameter_map["stream_num"].i = 10
    
    config.intra_op_parallelism_threads = 64  # 单操作内并行线程
    config.inter_op_parallelism_threads = 64  # 多操作间并行线程
    
    # 必须显式关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 显式关闭
        

    
    with tf.Session(config=config) as sess:
        # 进行集合通信初始化
        sess.run(npu_int)
        sfps.broadcast_embedding()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # 初始化AUC局部变量

        # 打印所有嵌入表的初始内容
        logger.info("=============================================part4==============================================================")    
        for idx, table in enumerate(all2all_embedding_tables):
            table_val = sess.run(table)
            logger.info(
                f"Initial embedding_table[{idx}] shape: {table_val.shape}, "
                f"mean: {table_val.mean():.4f}, std: {table_val.std():.4f}, "
                f"min: {table_val.min():.4f}, max: {table_val.max():.4f}"
            )
            # 打印表中的数据内容
            logger.info(f"Initial embedding_table[{idx}]  rows:\n{table_val}")


        sess.graph.finalize()
        tf.train.write_graph(sess.graph, path, 'train.pbtxt')

        before_train = time.time()
        for i in range(Iters):
            if i == 2:
                before_train = time.time()
            
            read_dataset_start = time.time()
            input_local_batch, input_labels = get_input(sfps.get_group_rank(), i, uplicate_rate)

            # 准备 local embedding 和标签
            feed_dict = {label_placeholder: input_labels}
            for placeholder, data in zip(local_input_placeholders, input_local_batch):
                feed_dict[placeholder] = data

            print(f"Iter {i}: read dataset cost {time.time() - read_dataset_start:.4f}s")

            start_train = time.time()
			
			# ---------- 获取运行时中间变量：fetch_list拼接 ----------
            fetch_list = [all2allUpdateEmbedding, loss, auc_value, auc_update_op,
                          batch_emb_mean, batch_emb_std, batch_emb_shape]
            # 添加所有嵌入表以便打印更新后的值
            for tbl in all2all_embedding_tables:
                fetch_list.append(tbl)
            # 添加所有引擎的中间调试张量
            for engine in embedding_engines:
                for key, tensor in engine.debug_tensors.items():
                    fetch_list.append(tensor)
                    
            
            results = sess.run(fetch_list, feed_dict=feed_dict)
            # 解析结果
            idx = 0
            _, loss_val, auc_val, _ = results[idx:idx+4]; idx += 4
            emb_mean, emb_std, emb_shape_val = results[idx:idx+3]; idx += 3
            # 表格数据
            num_tables = len(all2all_embedding_tables)
            table_vals = results[idx:idx+num_tables]; idx += num_tables
            # 剩下的都是各引擎的中间张量，按引擎顺序和 debug_tensors 字典顺序
            debug_results = {}
            for engine in embedding_engines:
                engine_debug = {}
                for key in engine.debug_tensors.keys():
                    engine_debug[key] = results[idx]; idx += 1
                debug_results[engine.table_id] = engine_debug
            # 打印常规信息
            logger.info(f"=============================================iter:{i}==============================================================")    
            logger.info(f"Iter {i}: loss={loss_val:.6f}, auc={auc_val:.6f}, "
                        f"batch_embedding shape={emb_shape_val}, "
                        f"batch_embedding mean={emb_mean:.4f}, std={emb_std:.4f}")

            # 打印所有嵌入表更新后的内容
            logger.info(f"=============================================iter:{i} embedding_table content==============================================================")    
            for j, tab_val in enumerate(table_vals):
                logger.info(f"Iter {i}: embedding_table[{j}] after update: "
                            f"mean={tab_val.mean():.4f}, std={tab_val.std():.4f}, "
                            f"min={tab_val.min():.4f}, max={tab_val.max():.4f}")
                logger.info(f"Iter {i}: embedding_table[{j}] rows:\n{tab_val}")
                
            # 打印每个引擎的中间变量
            logger.info(f"=============================================iter:{i} embedding_engines debug tensor==============================================================")   
            for engine in embedding_engines:
                tid = engine.table_id
                dbg = debug_results[tid]
                logger.info(f"Iter {i} Engine table_id={tid} forward/backward intermediates:")
                for k, v in dbg.items():
                    # 根据张量类型选择打印方式
                    if isinstance(v, np.ndarray):
                        logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                        logger.info(f"  {k}: {v}")
                    else:
                        logger.info(f"  {k}: {v}")
            
            logger.info(f"Iter {i}: step cost {time.time() - start_train:.4f}s (excluding read dataset)")

        sfps.barrier(4)
        total_time = time.time() - before_train
        logger.info(f"All steps finished, total training time: {total_time:.4f}s")
        time.sleep(0.1)

    print(f"Worker finished, total time since before_train: {time.time() - before_train:.4f}s")
    #sfps.shuts_down()
    print("python shutdown ")