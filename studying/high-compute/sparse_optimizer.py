from tensorflow.python.training import adam
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.python.ops.math_ops as math_ops
import math


class CustomizedLazyAdam(adam.AdamOptimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        use_locking=False,
        name="LazyAdam"
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            use_locking=use_locking,
            name=name
        )
        self._slot_num = 2  # Adam需2个slot：动量(m)、速度(v)

    def _create_slots(self, var_list):
        """为稀疏变量创建动量(m)、速度(v)的slot（优化器核心）"""
        if not var_list:
            return
        # beta1_power、beta2_power
        first_var = var_list[0]
        self._create_non_slot_variable(initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2, name="beta2_power", colocate_with=first_var)
        # 为每个嵌入表创建slot（初始值为0）
        for var in var_list:
            self._zeros_slot(var, "m", self._name + "/momentum")
            self._zeros_slot(var, "v", self._name + "/velocity")

    def _apply_sparse_shared(self, grad, var, indices, scatter_add_fn):
        """核心：稀疏梯度的Adam更新逻辑（动量+速度+变量更新）"""
        # 获取衰减系数累积值（beta1^t、beta2^t）
        beta1_power, beta2_power = self._get_beta_accumulators()
        # 类型统一
        var_dtype = var.dtype.base_dtype
        lr = math_ops.cast(self._lr_t, var_dtype)
        beta1 = math_ops.cast(self._beta1_t, var_dtype)
        beta2 = math_ops.cast(self._beta2_t, var_dtype)
        epsilon = math_ops.cast(self._epsilon_t, var_dtype)
        beta1_power = math_ops.cast(beta1_power, var_dtype)
        beta2_power = math_ops.cast(beta2_power, var_dtype)

        # 计算Adam学习率
        learning_rate = lr * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power)

        # 1. 更新动量m
        m_slot = self.get_slot(var, "m")
        old_m = tf.gather(m_slot, indices)
        new_m = beta1 * old_m + (1 - beta1) * grad
        m_update = scatter_add_fn(m_slot, tf.expand_dims(indices, 1), new_m - old_m)

        # 2. 更新速度v
        v_slot = self.get_slot(var, "v")
        old_v = tf.gather(v_slot, indices)
        new_v = beta2 * old_v + (1 - beta2) * math_ops.square(grad)
        v_update = scatter_add_fn(v_slot, tf.expand_dims(indices, 1), new_v - old_v)

        # 3. 更新嵌入表（稀疏更新）
        denom = math_ops.sqrt(new_v) + epsilon
        var_delta = -learning_rate * new_m / denom
        var_update = scatter_add_fn(var, tf.expand_dims(indices, 1), var_delta)

        return tf.group(m_update, v_update, var_update)  # 确保三操作同时完成

    def _apply_sparse(self, grad, var):
        """处理普通稀疏变量（IndexedSlices类型梯度）"""
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: tf.compat.v1.scatter_nd_add(x, i, v)
        )

    def _resource_apply_sparse(self, grad, handle, indices):
        """处理Resource类型稀疏变量（兼容NPU资源变量）"""
        def resource_scatter_add(x, i, v):
            with ops.control_dependencies([tf.raw_ops.ResourceScatterNdAdd(resource=x.handle, indices=i, updates=v)]):
                return x.value()
        return self._apply_sparse_shared(grad, handle, indices, resource_scatter_add)

    def _apply_dense(self, grad, var):
        raise NotImplementedError("仅支持稀疏变量更新（嵌入表）")


def create_hash_optimizer(learning_rate=0.1):
    return CustomizedLazyAdam(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        name="LazyAdam"
    )