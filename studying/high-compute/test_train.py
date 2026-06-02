import os
import time
import tensorflow as tf
import shutil
import fnmatch
import config
import numpy as np
from config import train_para

if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
# tf.disable_v2_behavior()
from tensorflow.python.framework import graph_util
import horovod.tensorflow as hvd
import psutil
from collections import defaultdict
from SFPS.tools.merge_increment import compare_sparse_file
from SFPS.tools.checkpoint_tools import merge_and_split_sparse_file
from SFPS.tools import get_sparse_header
from tensorflow.python.framework import ops
import tensorflow.python.ops.math_ops as math_ops 
import math
import sys
import npu_device
from npu_device.compat.v1.npu_init import *

from sparse_optimizer import *
from memory_profiler import log_memory, log_memory_banner, estimate_merged_tables_mb

npu_device.compat.enable_v1()

print(f"host ip addr {os.getenv('Worker_HOST_IP_ADDR')}")
print(f"prefetch step {int(os.getenv('prefetch_step', 0))}")
print(f"ec async {os.getenv('EC_ASYNC', False)}")
rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
print("======== rank id ========", rank)
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "0"

workerNum = int(os.environ.get('VC_WORKER_NUM', '0'))
print("workerNum:",  workerNum)


import random
seed = 1234
tf.compat.v1.set_random_seed(seed)
random.seed(seed)
np.random.seed(seed)

# general_slot =  [105, 1, 1, 127, 1, 122, 1, 98, 1, 124, 143, 1, 119, 132, 109, 1, 125, 1, 118, 1, 1, 1, 1, 1, 1, 1, 115,
#                 117, 123, 1, 1, 110, 121, 1, 118, 100, 105, 1, 1, 1, 1, 120, 125, 1, 130, 1, 1, 109, 1, 98, 1, 1, 1, 1,
#                 1, 125, 1, 116, 1, 121, 140, 1, 1, 109, 1, 127, 1, 1, 123, 1, 119, 1, 1, 1, 1, 1, 1, 118, 1, 1, 1, 113,
#                 131, 116, 1, 1, 108, 1, 108, 118, 112, 1000, 132, 1, 1, 1, 1, 1, 1, 1, 1, 132, 104, 110, 1, 1, 130, 114,
#                 139, 1, 1, 1, 1, 1, 1, 1, 1, 115, 119, 1, 128, 113, 108, 120, 1, 139, 1, 100, 1, 1, 119, 1, 143, 1, 1,
#                 114, 1, 121, 1, 1, 1, 1, 1, 142, 1, 1, 125, 1, 130, 1, 1, 1, 133, 1, 1, 1, 114, 106, 139, 1, 124, 103,
#                 108, 1, 1, 1, 1, 133, 1, 1, 1, 1, 1, 1, 1, 1, 124, 122, 124, 130, 128, 1, 1, 1, 1, 1, 129, 1, 116, 1,
#                 137, 111, 128, 1, 1, 1, 109, 1, 1, 1, 1, 1, 1, 1, 111, 1, 116, 91, 1, 1, 1, 111, 128, 1, 129, 1, 128, 1,
#                 1, 109, 1, 1, 1, 117, 1, 115, 1, 126, 1, 122, 107, 134, 1, 1, 114, 1, 1, 1, 1, 1, 1, 105, 1, 1, 1, 1, 1,
#                 122, 1, 1, 1, 118, 1, 104, 128, 119, 108, 130, 1, 120, 1, 122, 122, 1, 1, 1, 122, 133, 1, 1, 1, 1, 1, 1,
#                 132, 101, 1, 112, 104, 1, 121, 1, 1, 1, 1, 1, 116, 1, 1, 1, 1, 1, 138, 129, 1, 135, 110, 1, 137, 1, 1,
#                 1, 1, 1, 1, 1, 1, 1, 1, 131, 118, 1, 133, 118, 1, 1, 1, 120, 1, 1, 1, 1, 1, 121, 99, 107, 113, 129, 1,
#                 116, 129, 144, 1, 134, 109, 1, 1, 127, 1, 1, 1, 121, 102, 1, 120, 102, 131, 1, 1, 1, 1, 1, 1, 119, 1, 1,
#                 101, 1, 102, 1, 1, 1, 121, 122, 108, 107, 1, 1, 1, 132, 1, 1, 1, 120, 125, 134, 97, 1, 1, 1, 1, 107,
#                 115, 1, 1, 1, 1, 101, 1, 1, 1, 122, 112, 1, 1, 144, 1, 1, 117, 113, 1, 1, 1, 1, 128, 1, 1, 1, 1, 1, 1,
#                 1, 1, 1, 1, 102, 1, 116, 125, 1, 129, 1, 1, 105, 117, 1, 1, 104, 103, 1, 127, 1, 1, 121, 1, 1, 1, 1, 1,
#                 1, 129, 113, 116, 1, 115, 1, 1, 116, 1, 100, 1, 130, 1, 1, 108, 1, 121, 139, 138, 103, 1, 1, 1, 1, 115,
#                 122, 1, 1, 1, 1, 1, 123, 128, 1, 125, 1, 1, 1, 1, 107, 1, 1, 110, 1, 1, 1, 1, 1, 134, 123, 104, 129, 1,
#                 1, 112, 104, 110, 1, 128, 1]


# # general_slot = [50] * 50
# general_slot =  [1, 68, 1, 1, 1, 72, 1, 1, 1, 1000, 1, 59, 1, 81, 1, 1, 47, 1, 1, 93, 1, 1, 77, 1, 1, 62, 1, 1, 88, 1, 53, 1, 1, 98, 1, 1, 71, 1, 84, 1, 1, 65, 1, 1, 91, 1, 1, 80, 1, 1]




print("worker进程的进程号是:",  os.getpid())
# bs = int(os.environ.get('bacth_size', 10000))
# # all2all_slot = [50] * 50
# all2all_slot = general_slot
# total_slot = all2all_slot
# total_all2all_slot_size = sum(all2all_slot)
# embedding_dim = 8
# Iters = 30

# # uplicate_rate = 0.9
# v_size = 5000 #int(bs*(1-uplicate_rate)*50)
# local_vocabulary_size = [v_size] * len(all2all_slot)   # 重复度90%，bs*slot[i]

path = os.path.join(os.getcwd(), 'checkpoint')

bs = config.bs
seq_len = config.seq
# all2all_slot = [seq_len] * config.f_num
# all2all_slot = [
#   4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000,
#   4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000,
#   1206, 1206, 1206, 1206, 1206, 1206,
#   1154, 1154, 1154, 1154, 1154, 1154, 1154, 1154,
#   1153, 1153, 1153, 1153, 1153, 1153, 1153, 1153,
#   1153, 1153, 1153, 1153, 1153, 1153, 1153, 1153,
#   1153, 1153, 1153, 1153
# ]  # 103528

slot_num = config.slot_num
general_slot = config.general_slot
all2all_slot = config.general_slot[:slot_num]

seq_len = len(all2all_slot)

total_slot = all2all_slot
total_all2all_slot_size = sum(all2all_slot)
embedding_dim = config.embedding_dim
Iters = config.iters




num_npus =  workerNum                 # NPU卡数量

local_vocabulary_size = []
total_voc_size = 0
for i in range(len(all2all_slot)):
    local_vocabulary_size.append(config.single_vocabulary_size[i]//num_npus+1)
    total_voc_size += local_vocabulary_size[i]
conbiner = 1  # 求和

prefetch_step = 0
is_padding = False
padding_key = -1
is_completion = False
completion_key = 99
default_keys =  [-3] * len(total_slot)

# 学习率
LEARNING_RATE_SPARSE = 0.005   # 0.00l5
LEARNING_RATE_DENSE = 0.0001   # 0.0001
# 常量初始化器值
CONSTANT_INIT = 1
# 合表内子表数量上限
MAX_TABLES_PER_GROUP = 50
MAX_MERGE_TABLE_CAPACITY = 350000 * 1000
MAX_MERGE_TABLE_SLOT_SIZE = 4001
# # 是否启用池化模式
# use_pooling_in_pull = False
# # 是否运行合表测试的模式
# USE_MERGED_TABLE = os.environ.get('USE_MERGED_TABLE', 'true').lower() == 'true'
# # 单表多卡词汇表大小
# VOCABULARY_SIZE = 5000
# single_vocabulary_size = VOCABULARY_SIZE // num_npus + 1
# # v_size = int(bs*(1-uplicate_rate)*50)
# local_vocabulary_size = [single_vocabulary_size] * len(all2all_slot)  # 5000

# 是否运行合表测试的模式
USE_MERGED_TABLE = os.environ.get('USE_MERGED_TABLE', 'true').lower() == 'true'
# 全局开关：True 表示 pull 接口返回聚合特征，False 表示返回原始序列,非池化模式
use_pooling_in_pull = os.environ.get('use_pooling_in_pull', 'false').lower() == 'true'

print("-----------------开始 all2all 性能测试")
print(f"step_size: {Iters}")
print(f"batchsize: {bs}")
print(f"dim: {embedding_dim}")
print(f"seq_len: {seq_len}")
print(f"feature_num: {config.f_num}")
print(f"capacity: {config.cap}")
print(f"slot_num: {slot_num}")
print(f"use_pooling_in_pull: {use_pooling_in_pull}")
print(f"total_voc_size: {total_voc_size}")
print(f"all2all_slot: {all2all_slot}")
print("-------------------------------------")


# ===========================
# 合表功能实现
# ===========================

class SubTableConfig:
    """小表配置：用于合表功能"""
    def __init__(self, table_id, capacity, optimizer, key_offset=0, slot_size=1):
        self.table_id = table_id
        self.capacity = capacity  # 小表容量
        self.optimizer = optimizer  # 该小表使用的optimizer（可与其他小表不同）
        self.key_offset = key_offset  # 该小表在大表中的起始key偏移
        self.slot_size = slot_size  # slot大小


class MergedTable:
    """
    合表管理器：管理多个小表合并成的大表

    设计理念：
    - 多个小表在物理上合并成一个大表
    - 通过key_offset映射实现逻辑上的分表
    - 小表的input_ids通过加key_offset变成global_key
    - global_key在大表中查找对应的uindex
    """
    def __init__(self, table_id, embedding_dim, global_capacity, rank, sparse_optimizer):
        self.table_id = table_id
        self.embedding_dim = embedding_dim
        self.global_capacity = global_capacity  # 大表总容量
        self.rank = rank
        self.sparse_optimizer = sparse_optimizer

        # 创建大表 (tf.Variable)
        with tf.compat.v1.variable_scope(f"merged_table_scope_{rank}_table_{self.table_id}"):
            self.global_embedding_table = tf.compat.v1.get_variable(
                f'merged_embedding_table_{self.table_id}',
                shape=[global_capacity, embedding_dim],
                initializer=tf.constant_initializer(value=CONSTANT_INIT),
                # initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=1234),
                dtype=tf.float32
            )

        # 小表配置列表
        self.sub_tables = []  # List[SubTableConfig]

        # 小表到大表的embedding offset映射
        self.sub_table_offsets = []  # 每个小表在大表embedding中的起始offset
        self.sub_table_capacities = []  # 每个小表的capacity

        # 反向映射：用于调试和验证
        # self.global_key_to_sub_table = {}

    def add_sub_table(self, sub_table_config):
        """添加一个小表到合表"""
        # print(f"[MergedTable] merged_table_{self.table_id}, add sub_table_{sub_table_config.table_id}, "
            #   f"key_offset:{sub_table_config.key_offset}, slot_size:{sub_table_config.slot_size}, capacity:{sub_table_config.capacity}")
        self.sub_tables.append(sub_table_config)
        self.sub_table_offsets.append(sub_table_config.key_offset)
        self.sub_table_capacities.append(sub_table_config.capacity)

    def get_sub_table_embedding_range(self, sub_table_idx):
        """获取指定小表在大表中的embedding范围 [start, end)"""
        start = self.sub_table_offsets[sub_table_idx]
        end = start + self.sub_table_capacities[sub_table_idx]
        return start, end

    def get_num_sub_tables(self):
        """获取小表数量"""
        return len(self.sub_tables)

    def get_sub_tables(self):
        return self.sub_tables


class MergedTableKeyProcessor:
    """
    合表模式的key处理器

    核心功能：
    1. 将各小表的原始key转换为global_key（通过加key_offset）
    2. 拼接所有小表的global_key
    3. 执行_key_process获取uindex和indices
    4. 保存中间状态供后续反向传播使用
    """

    def __init__(self, merged_table, num_npus, _key_process_func):
        self.merged_table = merged_table
        self.num_npus = num_npus
        self._key_process = _key_process_func

        self.sub_table_ids = [st.table_id for st in merged_table.sub_tables]
        self.num_sub_tables = len(self.sub_table_ids)

        # 保存中间状态（用于反向传播）
        self.saved_context = None

    def process_all_sub_tables(self, input_ids_list):
        """
        处理所有小表的keys

        Args:
            input_ids_list: List[tf.Tensor] - 每个小表的input_ids

        Returns:
            combined_global_keys: tf.Tensor - 合并后的大表global keys
            sub_table_key_counts: List[int] - 每个小表的key数量
            send_sizes: tf.Tensor - 全局send_sizes [npu_num]
            recv_sizes: tf.Tensor - 全局recv_sizes [npu_num]
            indices: tf.Tensor - 全局indices (用于gather)
            uindex: tf.Tensor - 全局uindex
            offset_count: int - offset数量
            key_to_sub_table_map: tf.Tensor - key到大表的映射信息
        """
        # Step 1: 将每个小表的key转换为global_key
        global_keys_list = []
        sub_table_key_counts = []

        # TODO: for循环转换每个小表的key，性能较差，通过自定义op实现并行处理
        for i, input_ids in enumerate(input_ids_list):
            flat_input_ids = tf.reshape(input_ids, [-1])
            global_keys_list.append(flat_input_ids)

            # 记录该小表的key数量
            num_keys = tf.shape(flat_input_ids)[0]
            sub_table_key_counts.append(num_keys)

        # Step 2: 拼接所有小表的global_keys
        combined_global_keys = tf.concat(global_keys_list, axis=0)

        # Step 3: 执行_key_process (使用合并后的大表参数)
        
        send_sizes, recv_sizes, indices, uindex, offset_count = self._key_process(
            combined_global_keys,
            self.merged_table.table_id,
            slot_size=1,  # 已经是拼接后的，slot_size设为1
            name_=f'keyprocess_merged_{self.merged_table.table_id}',
            is_prefetch_=False,
            npu_num_=self.num_npus,
            cap_=self.merged_table.global_capacity
        )
        uindex = uindex[:offset_count]
        # TODO: 假设多卡数据分布均匀，如果超过均值，进行取模。后续优化
        uindex = uindex % (self.merged_table.global_capacity)

        # Step 4: 构建key到小表的映射信息
        # 用于后续将global embedding映射回各小表
        key_to_sub_table_map = self._build_key_sub_table_mapping(sub_table_key_counts)

        # 保存中间状态（用于反向传播）
        self.saved_context = {
            'combined_global_keys': combined_global_keys,
            'sub_table_key_counts': sub_table_key_counts,
            'send_sizes': send_sizes,
            'recv_sizes': recv_sizes,
            'indices': indices,
            'uindex': uindex,
            'offset_count': offset_count,
            'key_to_sub_table_map': key_to_sub_table_map,
            'global_keys_list': global_keys_list,
        }

        return (combined_global_keys, sub_table_key_counts,
                send_sizes, recv_sizes, indices, uindex, offset_count,
                key_to_sub_table_map)

    def _build_key_sub_table_mapping(self, sub_table_key_counts):
        """构建combined_keys中每个key属于哪个小表的信息

        Returns:
            tf.Tensor: shape=[total_keys], 每个元素表示对应key所属的小表索引
        """
        
        indices = tf.range(tf.shape(sub_table_key_counts)[0])
        # 使用 tf.repeat 直接生成 [0, 0, 1, 1, 1, 2]
        return tf.repeat(indices, sub_table_key_counts)

    def get_saved_context(self):
        """获取保存的中间状态（用于反向传播）"""
        return self.saved_context


class MergedTableEmbeddingEngine:
    """
    合表模式的embedding引擎

    整合了合表的前向传播和反向传播逻辑
    """

    def __init__(self, merged_table, num_npus, _key_process_func, sparse_optimizers=None, combiner=0, pooling=False):
        self.merged_table = merged_table
        self.num_npus = num_npus
        self.key_processor = MergedTableKeyProcessor(
            merged_table, num_npus, _key_process_func)
        self.combiner = combiner
        self.pooling = pooling

        # 每个小表可以使用不同的optimizer
        if sparse_optimizers is None:
            # 默认使用merged_table的optimizer
            self.sparse_optimizers = merged_table.sparse_optimizer
        else:
            self.sparse_optimizers = sparse_optimizers

    def lookup_and_gather(self, input_ids_list):
        """
        合表模式的前向传播

        Args:
            input_ids_list: List[tf.Tensor] - 每个小表的input_ids

        Returns:
            sub_table_embeddings: List[tf.Tensor] - 每个小表的embeddings (gather后)
            sub_table_outputs: List[tf.Tensor] - 每个小表的pooling输出
        """
        # Step 1: 处理所有小表的keys
        (combined_keys, sub_table_key_counts,
         send_sizes, recv_sizes, indices, uindex, offset_count,
         key_to_sub_table_map) = self.key_processor.process_all_sub_tables(input_ids_list)

        # 保存offset_count供后续使用
        self.offset_count = offset_count

        # Step 2: 大表embedding lookup
        global_embeddings = tf.nn.embedding_lookup(
            self.merged_table.global_embedding_table, uindex
        )

        # Step 3: Embedding all2all通信
        send_embedding_sizes = recv_sizes * self.merged_table.embedding_dim
        recv_embedding_sizes = send_sizes * self.merged_table.embedding_dim

        recv_embeddings = alltoallv_exchange_embeddings(
            global_embeddings,
            send_embedding_sizes,
            recv_embedding_sizes
        )

        recv_embeddings = tf.reshape(recv_embeddings, [-1, self.merged_table.embedding_dim])

        # Step 4: 恢复原始顺序
        restored_embeddings = tf.gather(recv_embeddings, indices)

        # Step 5: 按小表分割embeddings
        sub_table_embeddings = self._split_embeddings_by_sub_table(
            restored_embeddings, sub_table_key_counts)

        # Step 6: 各小表独立执行pooling
        sub_table_outputs = []
        for i, emb in enumerate(sub_table_embeddings):
            batch_size = tf.shape(input_ids_list[i])[0]
            final_restored = tf.reshape(emb, [batch_size, -1, self.merged_table.embedding_dim])

            output = None
            if self.pooling:
                # TODO: 当前仅按照一个槽来处理，如果多槽，需要修改。形状：[bs, 1, dim]
                slot_aggs = [tf.reduce_sum(final_restored, axis=1) if self.combiner == 0 else tf.reduce_mean(final_restored, axis=1)] 
                output = tf.stack(slot_aggs, axis=1)
            else:
                output = final_restored
            sub_table_outputs.append(output)

        return sub_table_embeddings, sub_table_outputs

    def _split_embeddings_by_sub_table(self, restored_embeddings, sub_table_key_counts):
        """将embeddings按小表分割"""
        sub_table_embeddings = []
        start_idx = 0

        for count in sub_table_key_counts:
            sub_emb = restored_embeddings[start_idx:start_idx + count]
            sub_table_embeddings.append(sub_emb)
            start_idx += count

        return sub_table_embeddings

    def backward(self, sub_table_grads):
        """
        合表模式的反向传播和optimizer更新

        正确的逻辑：
        1. 梯度all2all后通过gather恢复原始顺序
        2. 按小表分割得到未去重的梯度
        3. 对每个小表：
           - 使用indices作为segment_ids进行segment_sum（聚合相同unique key的梯度）
           - 将segment_sum的结果（按unique位置排列）与uindex结合，得到(embedding_offset, gradient)对
           - 过滤出属于该小表embedding范围的(embedding_offset, gradient)对
           - 构建SparseGrad，只更新该小表范围

        Args:
            sub_table_grads: List[tf.Tensor] - 每个小表输入的梯度

        Returns:
            update_ops: List[tf.Operation] - 各小表的更新操作
        """
        # Step 1: 获取之前保存的中间结果
        ctx = self.key_processor.get_saved_context()
        if ctx is None:
            raise ValueError("请先调用lookup_and_gather进行前向传播")

        sub_table_key_counts = ctx['sub_table_key_counts']
        send_sizes = ctx['send_sizes']
        recv_sizes = ctx['recv_sizes']
        indices = ctx['indices']  # indices[i] = combined_keys中位置i在unique后的位置
        uindex = ctx['uindex']    # uindex[j] = unique_keys[j]对应的embedding偏移

        # Step 2: 将各小表梯度拼接成大表格式
        global_grads_list = []
        for i, grad in enumerate(sub_table_grads):
            if isinstance(grad, tf.IndexedSlices):
                target_length = sub_table_key_counts[i] 
                target_shape = [target_length, self.merged_table.embedding_dim]
                
                sparse_grad = tf.zeros(target_shape, dtype=grad.values.dtype)
                
                sparse_indices = tf.expand_dims(grad.indices, axis=1)
                sparse_grad = tf.tensor_scatter_nd_update(
                    tensor=sparse_grad,
                    indices=sparse_indices,
                    updates=grad.values
                )
                global_grads_list.append(sparse_grad)
            else:
                global_grads_list.append(grad)
        combined_global_grads = tf.concat(global_grads_list, axis=0)

        # Step 3: 大表梯度进行本地聚合（卡内相同key的梯度求和）
        local_aggregated_grads = tf.math.unsorted_segment_sum(
            data=combined_global_grads,
            segment_ids=indices,
            num_segments=tf.reduce_sum(send_sizes)
        )
        after_partition_grads = tf.reshape(local_aggregated_grads, [-1, self.merged_table.embedding_dim])
        
        # Step 4: 梯度all2all（与前向方向相反）
        send_embedding_sizes = send_sizes * self.merged_table.embedding_dim
        recv_embedding_sizes = recv_sizes * self.merged_table.embedding_dim
        recv_global_grads = alltoallv_exchange_embeddings(
            after_partition_grads,
            send_embedding_sizes,
            recv_embedding_sizes
        )

        # Step 5: 接收端二次聚合（跨卡相同key的梯度求和）
        global_grad_list = tf.reshape(recv_global_grads, [-1, self.merged_table.embedding_dim])
        unique_uindex, idx_mapping = tf.unique(uindex)
        global_aggregated_grads = tf.math.unsorted_segment_sum(
			data=global_grad_list,
			segment_ids=idx_mapping,  
			num_segments=tf.shape(unique_uindex)[0]
		)

        # Step 6：优化器更新
        final_sparse_grads = ops.IndexedSlices(
            values=global_aggregated_grads,
            indices=unique_uindex,
            dense_shape=tf.shape(self.merged_table.global_embedding_table)
        )
        grad_var_pair = [(final_sparse_grads, self.merged_table.global_embedding_table)]
        update_op = self.sparse_optimizers.apply_gradients(grad_var_pair)
        return update_op


def compute_merged_table_groups(sub_tables, merged_config=None):
    """
    计算合表分组

    Args:
        sub_tables: List[SubTableConfig] - 所有待合表的小表
        max_table_capacity: 最大单个大表容量
        max_tables_per_group: 最大每组小表数量

    Returns:
        List[List[int]]: 分组结果，每组是小表索引列表
    """
    if merged_config is None:
        merged_config = {}

    max_table_capacity = merged_config.get('max_table_capacity', MAX_MERGE_TABLE_CAPACITY)
    max_tables_per_group = merged_config.get('max_tables_per_group', MAX_TABLES_PER_GROUP)
    max_table_slot_size = merged_config.get('max_table_slot_size', MAX_MERGE_TABLE_SLOT_SIZE)
    
    # TODO: 限制合表规则：判断初始化器相同、优化器算法参数相同的小表才能合并成一个大表
    # 按slot_size从小到大排序（优先合并小表）
    sorted_indices = sorted(range(len(sub_tables)), key=lambda i: sub_tables[i].slot_size)
    print(f"计算合表分组，首先按照capacity进行升序排序，结果：{sorted_indices}")

    groups = []
    current_group = []
    current_capacity = 0
    current_slot_size = 0

    for idx in sorted_indices:
        table = sub_tables[idx]
        new_capacity = current_capacity + table.capacity
        new_slot_size = current_slot_size + table.slot_size

        # 检查是否需要开新组
        if (new_capacity > max_table_capacity or
            len(current_group) >= max_tables_per_group or
            new_slot_size > max_table_slot_size) and current_group:
            current_group.sort()
            groups.append(current_group)
            current_group = []
            current_capacity = 0
            current_slot_size = 0

        current_group.append(idx)
        current_capacity += table.capacity
        current_slot_size += table.slot_size

    if current_group:
        current_group.sort()
        groups.append(current_group)

    return groups


def get_merged_table_groups(vocab_sizes, slot_sizes, sparse_optimizer,
                            merged_config=None):
    """
    获取合表分组

    Args:
        vocab_sizes: List[int] - 各小表vocab size
        slot_sizes: List[int] - 各小表slot大小
        sparse_optimizer: optimizer - 默认optimizer
        merged_config: dict - 合表配置

    Returns:
        groups: List[List[int]] - 分组结果，每组是小表索引列表
        sub_tables: List[SubTableConfig] - 所有原始小表配置列表
    """

    # 构建小表配置列表
    sub_tables = []
    for i in range(len(vocab_sizes)):
        sub_tables.append(SubTableConfig(
            table_id=i,
            capacity=vocab_sizes[i],
            optimizer=sparse_optimizer,  # 可以为不同小表指定不同optimizer
            key_offset=0,  # 暂时设为0，后面计算
            slot_size=slot_sizes[i]
        ))

    # 计算合表分组
    groups = compute_merged_table_groups(sub_tables, merged_config)
    
    # print(f"[MergedTable] Created {len(groups)} merged tables from {len(sub_tables)} sub tables")
    # for gi, group in enumerate(groups):
    #     print(f"  MergedTable {gi}: contains sub tables {group}, total capacity = {sum(sub_tables[i].capacity for i in group)}")
    return groups, sub_tables


def create_merged_embeddings(groups, sub_tables, embedding_dim, base_table_id, sparse_optimizer):
    """
    创建合表

    Args:
        groups: List[List[int]] - 分组结果，每组是小表索引列表
        sub_tables: List[SubTableConfig] - 所有原始小表配置列表
        embedding_dim: int - embedding维度
        base_table_id: int - 起始table_id
        sparse_optimizer: optimizer - 默认optimizer

    Returns:
        merged_tables: List[MergedTable] - 合并后的大表列表
        merged_table_embedding_engines: List[MergedTableEmbeddingEngine] - 合表引擎列表
        sub_table_to_merged_table: List[int] - 每个原始小表对应的大表索引
    """

    # 创建MergedTable
    merged_tables = []
    sub_table_to_merged_table = [-1] * len(sub_tables)

    for group_idx, group in enumerate(groups):
        # 计算大表总容量
        total_capacity = sum(sub_tables[i].capacity for i in group)

        # 创建MergedTable
        merged_table = MergedTable(
            table_id=base_table_id + len(merged_tables),
            embedding_dim=embedding_dim,
            global_capacity=total_capacity,
            rank=rank,
            sparse_optimizer=sparse_optimizer
        )

        # 添加小表到大表（分配key_offset）
        current_key_offset = 0
        for sub_idx in group:
            # 设置该小表的key_offset（小表的key会加这个偏移量变成global_key）
            sub_tables[sub_idx].key_offset = current_key_offset
            merged_table.add_sub_table(sub_tables[sub_idx])
            sub_table_to_merged_table[sub_idx] = len(merged_tables)
            current_key_offset += sub_tables[sub_idx].capacity

        merged_tables.append(merged_table)

    return merged_tables, sub_table_to_merged_table


def create_merged_all2all_embedding_for_every_slot(sfps, groups, sub_tables):
    """
        Note: create_embedding API should not be used mixedly with native api, like remote_embedding, xxx_embedding...
    """

    for group_idx, group in enumerate(groups):
        if len(group) < 1:
            print(f"Error: there is no sub_table in merged_table_{group_idx}")
            continue

        init = sfps.Initializer(1, 1, 'constant')
        lrs = sfps.ConstantLR(lr=0.005)
        opt = sfps.Adam(lrs, decay=0.0)
        comm_policy = sfps.communication_policy(communication_type='allreduce', grad_aggregation_type='average')

        default_key_config = default_keys[sub_tables[group[0]].table_id]
        feature_policy = sfps.feature_policy('counter_filter_with_default',
                                             counter_threshold=0,
                                             default_key=default_key_config, shrink_type='step',
                                             shrink_step_threshold=0)

        padding = sfps.padding_param(padding=is_padding, key=padding_key, mask=False)
        feature_completion_param = sfps.feature_completion_param(
            completion=is_completion,
            key=completion_key
        )

        merged_vocabulary_size = sum(sub_tables[idx].capacity for idx in group)
        merged_slot_size = sum(sub_tables[idx].slot_size for idx in group)
        sfps.create_table(sfps.c_lib.embedding_type.all2all, merged_vocabulary_size, embedding_dim,
                          bs, [merged_slot_size], opt, init, sfps.c_lib.key_type.int64, sfps.c_lib.pooling_type.sum,
                          sfps.c_lib.hash_type.hash, comm_policy, feature_policy, None)
        
        # print("hash_type=============:", sfps.c_lib.hash_type.hash)



def get_group_file_list(file_list, random_seed=None):
    groups = defaultdict(list)

    for file_path in file_list:
        try:
            filename = os.path.basename(file_path)
            name_without_ext = filename[:-len(".tfrecord")]
            _, num_part = name_without_ext.split("_", 1)
            n1, n2, n3 = num_part.split(".")

            key = (n1, n2)
            groups[key].append(file_path)
        except Exception:
            continue

    group_list = list(groups.values())

    if random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(group_list)

    return [f for group in group_list for f in group]

def parse_example(example_proto):
    feature_description = {
        "data": tf.io.FixedLenFeature([sum(general_slot)], tf.int64)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    example["data"] = tf.cast(example['data'], tf.int32)
    return example["data"]

def create_optimized_dataset(filenames):
    dataset = tf.data.Dataset.list_files(filenames, shuffle=False)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, buffer_size=3 * 1024 * 1024),
        cycle_length=8,
        block_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)

    # 添加错误忽略
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.bs, drop_remainder=True)

    if os.getenv("repeat") is not None:
        repeat_times = int(os.getenv("repeat", 0))
        if repeat_times == 0:
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(repeat_times)
        print(f"dataset read times {repeat_times}")

    dataset = dataset.prefetch(2)
    return dataset

def get_tf_input():
    train_dataset = []
    for root, dirs, files in os.walk(config.dataset_path):
        for file in files:
            if fnmatch.fnmatch(file, f'dataset_*.*.*.tfrecord'):
                train_dataset.append(os.path.join(config.dataset_path, file))
    base_seed = 42
    train_dataset = get_group_file_list(train_dataset, base_seed + sfps.get_group_rank())

    print(f"train_dataset size {len(train_dataset)}")
    # print(f"train_dataset {train_dataset}")

    datasets = create_optimized_dataset(train_dataset)
    iterator = datasets.make_one_shot_iterator()
    next_element = iterator.get_next()

    return next_element


def create_hash_optimizer(learning_rate=0.1):
    return CustomizedLazyAdam(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        name="LazyAdam"
    )

def dcn_model(batch_embedding, label, is_pooling=False):
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
    # ===== 1. Cross Network 修正 =====
    cross_depth = 3
    x0 = xl = output
    for i in range(cross_depth):
        # 关键修正：tf.compat.v1.glorot_normal_initializer()
        w = tf.compat.v1.get_variable(f'cross_w_{i}', shape=[input_dim, 1],
                            initializer=tf.compat.v1.glorot_normal_initializer())
        # 统一风格：tf.compat.v1.zeros_initializer()（虽tf.zeros_initializer在TF2也存在，但建议一致）
        b = tf.compat.v1.get_variable(f'cross_b_{i}', shape=[input_dim],
                            initializer=tf.compat.v1.zeros_initializer())
        xl = x0 * tf.matmul(xl, w) + xl + b
    # ===== 2. Deep Network 修正 =====
    deep_dims = [512, 128]
    x_deep = output
    for i, dim in enumerate(deep_dims):
        # 关键修正：tf.compat.v1.glorot_normal_initializer()
        w = tf.compat.v1.get_variable(f'deep_w_{i}', shape=[x_deep.shape[-1], dim],
                            initializer=tf.compat.v1.glorot_normal_initializer())
        # 统一风格：tf.compat.v1.zeros_initializer()
        b = tf.compat.v1.get_variable(f'deep_b_{i}', shape=[dim],
                            initializer=tf.compat.v1.zeros_initializer())
        x_deep = tf.nn.relu(tf.matmul(x_deep, w) + b)
    # 后续代码不变...
    concat = tf.concat([xl, x_deep], axis=1)
    logits = tf.compat.v1.layers.dense(concat, 1, activation=None)
    prediction = tf.identity(logits, name='prediction_output')
    loss = tf.reduce_mean(tf.square(logits - label), name='loss')
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


def batch_auc(labels, predictions):
    """
    计算单个 batch 的 ROC AUC（基于排序的近似）
    labels: [batch_size, 1]  0/1 标签
    predictions: [batch_size, 1]  预测概率
    """
    # 扁平化并降序排序
    labels = tf.cast(labels, tf.float32)
    predictions = tf.cast(predictions, tf.float32)
    
    # 将 labels 和 predictions 按预测值排序
    sorted_indices = tf.argsort(predictions, axis=0, direction='DESCENDING')
    sorted_labels = tf.gather(labels, sorted_indices[:, 0])
    
    # 正样本总数
    pos = tf.reduce_sum(sorted_labels)
    neg = tf.cast(tf.shape(sorted_labels)[0], tf.float32) - pos
    
    # 累积负样本数（每个正样本之前的负样本数）
    cum_neg = tf.cumsum(1.0 - sorted_labels)
    
    # AUC = (正样本排序得分之和 - 正样本最小可能排名和) / (pos * neg)
    # 使用标准公式：AUC = Σ(rank(pos_i)) - pos*(pos+1)/2 / (pos*neg)
    # 但简化为：正样本对应的累积负样本数之和 / (pos * neg)
    auc = tf.reduce_sum(sorted_labels * cum_neg) / (pos * neg + 1e-10)
    
    return auc



if "worker" == os.getenv('DMLC_ROLE', "worker"):
    from SFPS.tensorflow.TFWorker import TFWorker
    from SFPS.tensorflow.ops import _key_process

    log_memory_banner('startup (imports done)')
    log_memory('00_startup')
    
    print("\n" + "="*60)
    print("开始合表功能集成测试")
    print("="*60)
    print("\n1. 创建合表")
    
    # 合表配置
    sparse_optimizer = create_hash_optimizer(learning_rate=LEARNING_RATE_SPARSE)
    merged_config = {
        'max_table_capacity': MAX_MERGE_TABLE_CAPACITY,  # 大表最大容量
        'max_tables_per_group': MAX_TABLES_PER_GROUP,   # 每组最大小表数
        'max_table_slot_size': MAX_MERGE_TABLE_SLOT_SIZE
    }

    # 获取合表分组
    groups, sub_tables = get_merged_table_groups(
        local_vocabulary_size, all2all_slot, sparse_optimizer, merged_config=merged_config)
    log_memory('01_after_merge_groups',
               extra={'num_groups': len(groups), 'num_sub_tables': len(sub_tables)})

    sfps = TFWorker()
    sfps.prefetch(prefetch_step)
    create_merged_all2all_embedding_for_every_slot(sfps, groups, sub_tables)
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()
    sfps.total_embedding_count = len(sfps.table_create_infos)
    # sfps.total_embedding_count = 0
    #init_op = tf.global_variables_initializer()
    sfps.barrier()
    log_memory('02_after_npu_init_barrier')
    
    # 创建合表
    merged_tables, sub_table_to_merged = create_merged_embeddings(
        groups,
        sub_tables,
        embedding_dim,
        sfps.total_embedding_count,
        sparse_optimizer
    )
    #print(f"   创建了 {len(merged_tables)} 个大表")
    #print(f"   小表到大表的映射: {sub_table_to_merged}")
    _est_rows, _est_emb_mb = estimate_merged_tables_mb(merged_tables, embedding_dim)
    log_memory('03_after_create_merged_tables_graph_vars',
               merged_tables=merged_tables, embedding_dim=embedding_dim,
               extra={'num_merged_tables': len(merged_tables),
                      'est_embedding_only_mb': round(_est_emb_mb, 1),
                      'total_voc_size': total_voc_size})

    # 为每个大表创建MergedTableEmbeddingEngine
    merged_engines = []
    for mt in merged_tables:
        engine = MergedTableEmbeddingEngine(
            merged_table=mt,
            num_npus=num_npus,
            _key_process_func=_key_process,
            sparse_optimizers=create_hash_optimizer(learning_rate=LEARNING_RATE_SPARSE),
            combiner=conbiner,
            pooling=use_pooling_in_pull
        )
        merged_engines.append(engine)
    log_memory('04_after_build_engines')

    print("\n2. 创建input placeholders")
	# 标签占位符，替代固定 label
    label_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[bs, 1], name='label_placeholder')
	
    local_input_placeholders = [
        tf.compat.v1.placeholder(tf.int64, shape=[bs, None], name=f'input_placeholder_{i}')
        for i in range(len(all2all_slot))
    ]
    
    print("\n3. 构建合表前向传播图")
    # 按大表分组进行前向传播
    all2all_embeddings = [None] * len(all2all_slot)
    all2all_embeddings_before_pooling = [None] * len(all2all_slot)
    for eng_idx, engine in enumerate(merged_engines):
        # 找出属于这个大表的小表的input placeholders
        sub_table_indices = [i for i, mapped_idx in enumerate(sub_table_to_merged) if mapped_idx == eng_idx]
        sub_inputs = [local_input_placeholders[i] for i in sub_table_indices]

        #print(f"   大表{eng_idx}: 包含小表 {sub_table_indices}, 使用对应的input placeholders")

        # 前向传播
        embs_before_pooling, embs = engine.lookup_and_gather(sub_inputs)

        # 按照all2all_slot顺序组装
        for idx, emb_before_pooling, emb in zip(sub_table_indices, embs_before_pooling, embs):
            all2all_embeddings_before_pooling[idx] = emb_before_pooling
            all2all_embeddings[idx] = emb
    all2all_embedding = tf.concat(all2all_embeddings, axis=1)
    batch_embedding = all2all_embedding
    # 构建model
    print(f"batch_embedding static shape: {batch_embedding.shape}")
    loss, logits = dcn_model(batch_embedding, label_placeholder, is_pooling=use_pooling_in_pull)
    # loss = model(batch_embedding, label)
    
	# 预测概率，用于AUC计算
    pred_prob = tf.sigmoid(logits, name='pred_prob')

    # AUC 指标（累计）
    # auc_value, auc_update_op = tf.compat.v1.metrics.auc(labels=label_placeholder, predictions=pred_prob)
    
    auc_value = batch_auc(label_placeholder, pred_prob)

    # ADDED LOGGER: 定义用于监控的张量
    batch_emb_mean = tf.reduce_mean(batch_embedding)
    batch_emb_std = tf.math.reduce_std(batch_embedding)
    batch_emb_shape = tf.shape(batch_embedding)
    
    print(f"\n4. 模型输出shape: {all2all_embedding.shape}")
    log_memory('05_after_build_forward_graph', merged_tables=merged_tables, embedding_dim=embedding_dim)
    print("\n5. 构建反向传播图")
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE_DENSE)
    var_list = tf.compat.v1.trainable_variables()
    # print(f"-----var_list: {var_list}")
    dense_vars = [v for v in var_list
                  if not any(prefix in v.name for prefix in ['merged_', 'embedding'])]
    # print(f"-----dense_vars: {dense_vars}")
    dense_grads = optimizer.compute_gradients(loss, dense_vars)
    sparse_grads_for_backward = sparse_optimizer.compute_gradients(loss, all2all_embeddings_before_pooling)
    
    # # dense 层allreduce
    grads_and_vars = dense_grads
    grads_and_vars = allreduce(grads_and_vars)
    train_op = optimizer.apply_gradients(grads_and_vars)

    # 构建合表反向传播
    all_merged_update_ops = []
    with tf.control_dependencies([train_op]):
        for eng_idx, engine in enumerate(merged_engines):
            # 获取属于这个大表的sparse grads
            sub_table_indices = [i for i, mapped_idx in enumerate(sub_table_to_merged) if mapped_idx == eng_idx]
            sub_grads = [sparse_grads_for_backward[i][0] for i in sub_table_indices]

            # 反向传播
            update_ops = engine.backward(sub_grads)
            all_merged_update_ops.append(update_ops)

    all_merged_update_ops = tf.compat.v1.group(all_merged_update_ops, name="merged_train_step_group")

    next_dataset = get_tf_input()

    print(f"   反向传播图构建完成")
    log_memory('06_after_build_full_graph_backward',
               merged_tables=merged_tables, embedding_dim=embedding_dim,
               extra={'use_pooling': use_pooling_in_pull, 'bs': bs,
                      'total_all2all_slot_size': total_all2all_slot_size,
                      'total_keys_per_step': bs * total_all2all_slot_size})
    print("\n6. 准备训练")
    sfps.train()
    log_memory('07_after_sfps_train')
    
    print("\n6-2. after train ==============")
    
    #创建session，如果原始网络中使用了tf.device相关代码，则需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
    config = tf.compat.v1.ConfigProto()
    # 修正：OptimizerOptions 属于 TF1 兼容 API
    config.graph_options.optimizer_options.opt_level = tf.compat.v1.OptimizerOptions.L0
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = True

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 显式关闭
    
    #创建session，如果原始网络中使用了tf.device相关代码，则需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
    # config = tf.ConfigProto()
    # # config.gpu_options.visible_device_list = str(get_local_rank_id())
    # config.graph_options.optimizer_options.opt_level = tf.compat.v1.OptimizerOptions.L0
    # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name = "NpuOptimizer"
    # custom_op.parameter_map["mix_compile_mode"].b = True
    
    # # 必须显式关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
    # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 显式关闭
    # config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 显式关闭
        
    
    with tf.compat.v1.Session(config=config) as sess:
        log_memory('08_session_created')

        # 进行集合通信初始化
        print("000000000000000")
        sess.run(npu_int)
        log_memory('09_after_sess_run_npu_int')
        print("00000000000000011111111")
        sfps.broadcast_embedding()
        log_memory('09b_after_broadcast_embedding')
        print("00000000000000011111111222222")
        sess.run(tf.compat.v1.global_variables_initializer())
        log_memory('10_after_global_variables_initializer',
                   merged_tables=merged_tables, embedding_dim=embedding_dim,
                   extra={'note': 'embedding + Adam m/v slots materialized on NPU here'})
        # sess.run(tf.compat.v1.local_variables_initializer())  # 初始化AUC局部变量
        print("00000000000000011111111222222333333333")
        sess.graph.finalize()
        tf.compat.v1.train.write_graph(sess.graph, path, 'merged_train.pbtxt')
        log_memory('11_after_graph_finalize')
        print("00000000000000011111111222222333333333444444444444")
        loss_list = []
        
        # sfps.barrier()
        
        print("\n7. 开始训练循环")
        log_memory_banner('training loop')
        _profile_step_detail = os.environ.get('MEM_PROFILE_STEP_DETAIL', '1').lower() in ('1', 'true', 'yes')
        _profile_last_step = os.environ.get('MEM_PROFILE_LAST_STEP', '1').lower() in ('1', 'true', 'yes')
        _profile_every_n = int(os.environ.get('MEM_PROFILE_EVERY_N', '0'))
        before_train = time.time()
        for i in range(Iters):
            if i == 2:
                before_train = time.time()
            
            read_dataset_start = time.time()

            input_local_batch = sess.run(next_dataset)
            if i == 0 or i == Iters - 1:
                log_memory('iter%d_after_read_dataset' % i)
            print(f"input_local_batch 形状 = {input_local_batch.shape}")
            
            # input_local_batch = get_input(sfps.get_group_rank(), i, uplicate_rate)
            
            # 随机生成二分类标签（0或1），与AUC指标对应
            input_labels = np.random.randint(0, 2, size=(bs, 1)).astype(np.float32)
            
            # 准备 local embedding
            # 按  分割特征，axis=1 按列分割
            split_points = np.cumsum(general_slot)[:-1]  # 累计和去掉最后一个
            input_batches_split = np.split(input_local_batch, split_points, axis=1)
            # 每个元素形状变为 (bs, slot_i)

            count = 0
            for slot_input in input_batches_split:
                print(f"前10个input data size: {slot_input.shape}")
                count += 1
                if count > 10:
                    break
                # print(f"input data : {slot_input}")                

            feed_dict = {label_placeholder: input_labels}
            for placeholder, data in zip(local_input_placeholders[:slot_num], input_batches_split[:slot_num]):
                feed_dict[placeholder] = data

            # feed_dict = {label_placeholder: input_labels}
            # for placeholder, data in zip(local_input_placeholders, input_local_batch):
            #     feed_dict[placeholder] = data
            
            print(f"iter {i}, read dataset cost {time.time() - read_dataset_start}")

            start_train = time.time()
            _should_log_step = (i == 0 or i == Iters - 1
                                or (_profile_every_n > 0 and i % _profile_every_n == 0))
            if _should_log_step:
                log_memory('iter%d_before_run' % i)

            _split_this_step = ((i == 0 and _profile_step_detail)
                                or (i == Iters - 1 and _profile_last_step))
            if _split_this_step:
                # 拆分：前向+loss → 反向+dense/sparse更新，定位峰值来源
                _loss = sess.run(loss, feed_dict=feed_dict)
                log_memory('iter%d_after_forward_loss_only' % i,
                           extra={'loss': float(_loss)})
                sess.run(all_merged_update_ops, feed_dict=feed_dict)
                log_memory('iter%d_after_backward_sparse_dense_update' % i)
                emb_mean, emb_std, emb_shape_val, auc_val = sess.run(
                    [batch_emb_mean, batch_emb_std, batch_emb_shape, auc_value],
                    feed_dict=feed_dict)
            else:
                run_ops = [
                    loss,
                    all_merged_update_ops,
                    batch_emb_mean,
                    batch_emb_std,
                    batch_emb_shape,
                    auc_value
                ]
                _loss, _, emb_mean, emb_std, emb_shape_val, auc_val = sess.run(
                    run_ops, feed_dict=feed_dict)
                if _should_log_step:
                    log_memory('iter%d_after_full_train_step' % i,
                               extra={'loss': float(_loss)})

            print(f"11111111: rank_{sfps.get_group_rank()}, step_{i},"
                  f"batch_embedding shape={emb_shape_val}, mean={emb_mean:.4f}, std={emb_std:.4f},"
                  f"_loss={_loss} auc={auc_val}")
            
            # _ = sess.run([all_merged_update_ops], feed_dict=feed_dict)
            print(f"iter {i}, step cost {time.time() - start_train} exclude read dataset")
            print(f"iter {i}, step cost {time.time() - read_dataset_start} include read dataset")
            loss_list.append(_loss)
            
        avg_loss = np.mean(loss_list)
        print(f"Average loss: {avg_loss:.4f} Final AUC: {auc_val:.4f}")
        log_memory('12_after_all_iters', merged_tables=merged_tables, embedding_dim=embedding_dim)

        sfps.barrier()
        log_memory('13_after_barrier')
        
        print(f"all step train finished,cost {time.time() - before_train}")

        
        time.sleep(0.1)
        log_memory('13b_after_sleep_before_session_exit')

    print(f"worker finish time = {time.time() - before_train}")
    log_memory('14_after_session_exit')
    sfps.shuts_down()
    log_memory('15_after_sfps_shutdown')
    print("python shutdown ")