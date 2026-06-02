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

tf.enable_control_flow_v2()
device_ids = os.environ.get('NPU_IDS', '0,1,2,3,4,5,6,7').split(" ")  # 确保2个设备

rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
os.environ['ASCEND_DEVICE_ID'] = device_ids[rank]
os.environ['WORKER_DEVICE_ID'] = device_ids[rank]


from logger_utils import (
    setup_logger, log_section, log_array, log_config,
    log_feed_inputs, log_trace_results, log_model_outputs,
)
logger = setup_logger(rank)
_log_dir = os.environ.get('LOG_DIR', './logs')
logger.info(f'rank={rank} 日志文件: {os.path.join(_log_dir, f"training_rank_{rank}.log")}')
print(f"start rank: {rank}, log: {os.path.join(_log_dir, f'training_rank_{rank}.log')}")


import random
seed = 1234
tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)

general_slot =  [1, 1, 4000, 32, 1, 1, 1, 1, 54, 1, 94, 13, 100, 1, 8, 25, 19, 1, 24, 16, 1, 36, 1, 1, 36, 1, 1,
                4000, 1, 1, 1, 1, 18, 1, 19, 1, 1, 2, 2, 38, 1, 1, 1, 1, 6, 47, 1, 1, 1, 1, 1, 25, 44, 1, 1, 26, 1,
                1, 125, 1, 1, 1, 1, 1, 25, 57, 1, 6, 30, 1, 1, 12, 4, 1, 31, 44, 43, 1, 35, 1, 5, 11, 39, 47, 1, 1,
                16, 1, 73, 1, 5, 1, 34, 1, 3, 1, 26, 1, 1, 1, 1, 1, 1, 1, 14, 15, 1, 1, 1, 9, 3, 10, 1, 1, 5, 1, 1,
                1, 45, 1, 16, 1, 1, 17, 30, 1, 30, 1, 1, 4, 1, 1, 14, 1, 1, 1, 1, 1, 1, 1, 4, 1, 22, 29, 1, 42, 1,
                11, 1, 4000, 4000, 127, 5, 1, 18, 26, 1, 30, 1, 1, 1, 1, 1, 35, 1, 1, 32, 1, 1, 1, 1, 1, 5, 14, 1,
                1, 1, 1, 34, 1, 57, 1, 11, 11, 1, 1, 1, 1, 1, 1, 22, 1, 1, 29, 1, 1, 1, 1, 1, 10, 1, 119, 1, 45, 21,
                1, 1, 83, 41, 1, 1, 1, 63, 1, 1, 1, 1, 2, 1, 1, 61, 1, 1, 2, 11, 9, 1, 1, 19, 1, 1, 1, 1, 48, 4, 1,
                1, 1, 8, 1, 1, 42, 4000, 1, 1, 1, 1, 6, 1, 10, 60, 79, 134, 1, 5, 1, 1, 1, 1, 4, 1, 94, 14, 45, 8,
                16, 1, 8, 1, 1, 1, 1, 1, 1, 1, 3, 113, 1, 1, 1, 1, 18, 37, 1, 1, 1, 1, 60, 18, 7, 1, 1, 1, 1, 11, 1,
                1, 1, 4, 20, 1, 101, 1, 4, 4000, 28, 15, 1, 1, 1, 7, 1, 58, 1, 1, 1, 50, 1, 2, 190, 1, 1, 12, 1,
                136, 1, 66, 22, 1, 1, 87, 1, 1, 1, 1, 45, 1, 12, 1, 1, 1, 1, 1, 1, 1, 73, 1, 1, 1, 1, 4, 4000, 15,
                4000, 32, 21, 1, 1, 3, 22, 1, 1, 1, 121, 1, 1, 1, 55, 1, 18, 1, 1, 1, 1, 1, 1, 1, 3, 33, 29, 1, 1,
                36, 1, 1, 43, 54, 1, 6, 1, 1, 1, 6, 2, 17, 1, 1, 24, 11, 1, 12, 1, 37, 10, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 16, 1, 1, 68, 1, 1, 59, 1, 1, 1, 1, 1, 21, 19, 1, 1, 15, 1, 1, 1, 15, 48, 1, 9, 1, 1, 1,
                59, 4, 75, 5, 4000, 18, 25, 2, 47, 1, 2, 1, 1, 11, 1, 1, 1, 1, 9, 1, 1, 30, 1, 34, 1, 1, 1, 30, 1,
                1, 1, 1, 1, 1, 1, 126, 1, 32, 1, 1, 15, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 22, 1, 1, 6, 1, 1, 1, 12, 1,
                83, 1, 81, 8, 1, 1, 26, 1, 1, 64, 1, 13, 1, 22, 61, 1, 1, 18, 1, 16, 1, 1, 1, 7, 4000, 21, 8, 23,
                31, 1, 1, 1, 1, 1, 6, 1, 4, 19, 1, 1, 1, 1, 9, 1, 1, 10, 1, 39, 1, 1, 1, 1, 56, 1, 1, 1, 15, 1, 1,
                85, 1, 1, 1, 1, 12, 44, 1, 1, 1, 10, 136, 13, 1, 1, 2, 5, 34, 1, 1, 1, 1, 1, 1, 20, 9, 13, 1, 1, 4,
                19, 1, 82, 1, 1, 1, 19, 9, 1, 11, 1, 69, 1, 1, 1, 1, 1, 1, 12, 1, 46, 54, 1, 4, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 14, 2, 1, 11, 1, 1, 27, 1, 1, 1, 1, 1, 1, 1, 1, 37, 1, 4, 16, 1, 1, 5, 61, 1, 14, 16,
                43, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 1, 9, 73, 55, 1, 1, 15, 1, 1, 41, 1, 1, 1, 4000, 31, 1, 1, 18,
                1, 164, 1, 1, 1, 19, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 22, 64, 1, 1, 48, 42, 2, 1, 1, 55, 3, 1, 1, 8, 1,
                1, 1, 1, 1, 1, 14, 7, 1, 1, 1, 1, 1, 4000, 11, 46, 18, 1, 51, 5, 1, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 7, 41, 1, 1, 32, 63, 3, 1, 13, 1, 34, 1, 1, 61, 1, 1, 2, 1, 1, 10, 1, 1, 1, 1, 1, 187, 1, 1,
                1, 5, 1, 1, 1, 1, 1, 4, 1, 27, 5, 30, 1, 1, 1, 1, 1, 1, 1, 35, 1, 29, 1, 1, 1, 11, 1, 1, 1, 1, 1,
                25, 1, 23, 78, 1, 1, 1, 1, 1, 21, 5, 1, 25, 1, 1, 1, 4000, 76, 1, 6, 1, 1, 1, 22, 1, 13, 1, 1, 1, 1,
                66, 1, 39, 53, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 82, 6, 21, 1, 11, 11, 1, 1, 1, 17, 11, 4000,
                22, 1, 1, 1, 33, 1, 1, 1, 23, 1, 24, 1, 12, 1, 1, 1, 19, 1, 1, 1, 1, 1, 1, 1, 38, 1, 1, 1, 11, 1, 1,
                29, 14, 50, 1, 40, 7, 1, 47, 27, 1, 1, 19, 15, 1, 1, 69, 1, 1, 1, 1, 40, 29, 1, 1, 1, 1, 1, 1, 2, 1,
                1, 1, 57, 3, 1, 1, 1, 1, 1, 31, 21, 21, 1, 5, 1, 1, 47, 26, 1, 20, 6, 1, 13, 1, 1, 4000, 16, 1, 1,
                1, 10, 1, 17, 1, 1, 1, 1, 1, 1, 1, 1, 18, 1, 10, 1, 1, 1, 13, 1, 1, 1, 1, 1, 79, 15, 1, 8, 21, 1, 5,
                4000, 21, 1, 72, 1, 16, 1, 1, 1, 23, 37, 1, 1, 1, 1]

slots_vocabulary_size = [1948, 64, 1580735, 899507, 1675, 701, 1695, 1896, 1517918, 258, 2642302, 365425, 2810959, 318, 224876, 702740, 534082, 811, 674630, 449753, 242, 1011945, 1914, 1342, 1011945, 26, 1071, 3567150, 101, 1000, 1561, 1589, 505972, 544, 534082, 1739, 1042, 56219, 56219, 1068164, 743, 1711, 1575, 1429, 168657, 1321151, 1259, 1896, 612, 913, 1899, 702740, 1236822, 1953, 205, 730849, 1223, 741, 3513699, 1911, 1465, 764, 586, 593, 702740, 1602247, 1444, 168657, 843287, 69, 1202, 337314, 112438, 205, 871398, 1236822, 1208712, 1702, 983836, 1799, 140547, 309205, 1096274, 1321151, 969, 1323, 449753, 159, 2052000, 1610, 140547, 987, 955726, 867, 84328, 68, 730849, 1364, 1314, 402, 1410, 810, 403, 1873, 393534, 421643, 1011, 472, 782, 252986, 84328, 281096, 1395, 1440, 140547, 619, 1451, 246, 1264931, 662, 449753, 1882, 1946, 477863, 843287, 938, 843287, 973, 1769, 112438, 57, 1792, 393534, 1374, 375, 230, 180, 1892, 1978, 793, 112438, 740, 618411, 815178, 1396, 1180603, 514, 309205, 5, 2580142, 4143325, 3569919, 140547, 400, 505972, 730849, 320, 843287, 867, 1586, 1337, 1893, 1496, 983836, 1709, 727, 899507, 897, 1048, 1478, 459, 1885, 140547, 393534, 1346, 534, 1534, 756, 955726, 296, 1602247, 1864, 309205, 309205, 881, 1986, 779, 1778, 603, 221, 618411, 1551, 1466, 815178, 1596, 784, 1745, 1067, 1313, 281096, 21, 3345042, 1105, 1264931, 590301, 1827, 1347, 2333096, 1152493, 172, 269, 1683, 1770904, 1520, 557, 878, 832, 56219, 876, 758, 1714685, 1485, 192, 56219, 309205, 252986, 401, 472, 534082, 1383, 14, 1847, 1859, 1349260, 112438, 1871, 1383, 1856, 224876, 1939, 152, 1180603, 996843, 832, 1698, 785, 422, 168657, 19, 281096, 1686576, 2220658, 3766686, 1014, 140547, 258, 843, 829, 1939, 112438, 364, 2642302, 393534, 1264931, 224876, 449753, 632, 224876, 1331, 268, 306, 1675, 1165, 1138, 1958, 84328, 3176384, 1607, 359, 787, 772, 505972, 1040054, 823, 372, 1753, 731, 1686576, 505972, 196767, 7, 627, 1257, 1417, 309205, 1430, 399, 33, 112438, 562192, 937, 2839069, 584, 112438, 4225230, 787069, 421643, 105, 1605, 1569, 196767, 230, 1630356, 581, 147, 1292, 1405480, 1364, 56219, 5340823, 973, 1444, 337314, 910, 3822905, 1151, 1855233, 618411, 1958, 1139, 2445535, 896, 208, 67, 740, 1264931, 1646, 337314, 74, 267, 970, 1268, 731, 1902, 1183, 2052000, 782, 1880, 1326, 173, 112438, 3930936, 421643, 3327557, 899507, 590301, 1278, 1689, 84328, 618411, 683, 1644, 122, 3401261, 1901, 1346, 1391, 1546027, 820, 505972, 681, 328, 279, 676, 1974, 1832, 1654, 84328, 927616, 815178, 725, 324, 1011945, 498, 959, 1208712, 1517918, 665, 168657, 8, 164, 337, 168657, 56219, 477863, 413, 1933, 674630, 309205, 1540, 337314, 582, 1040054, 281096, 1340, 584, 869, 1310, 1641, 1527, 1693, 231, 1905, 463, 119, 449753, 429, 1390, 1911453, 1550, 671, 1658466, 344, 1499, 1505, 1799, 1592, 590301, 534082, 899, 1917, 421643, 1622, 263, 992, 421643, 1349260, 372, 252986, 1290, 423, 731, 1658466, 112438, 2108220, 140547, 558524, 505972, 702740, 56219, 1321151, 1018, 56219, 301, 926, 309205, 1188, 1634, 698, 1412, 252986, 1327, 1631, 843287, 1281, 955726, 2, 1233, 1253, 843287, 47, 1271, 1833, 1177, 1480, 286, 1096, 3541809, 1107, 899507, 1100, 221, 421643, 1555, 308, 894, 572, 1959, 855, 252986, 596, 420, 1409, 618411, 1735, 1002, 168657, 1641, 294, 1707, 337314, 1148, 2333096, 1806, 2276877, 224876, 217, 131, 730849, 340, 1927, 1799014, 1087, 365425, 1477, 618411, 1714685, 967, 135, 505972, 1842, 449753, 1580, 1700, 1688, 196767, 2887927, 590301, 224876, 646520, 871398, 762, 1204, 1938, 759, 1458, 168657, 1669, 112438, 534082, 1869, 1785, 350, 298, 252986, 827, 153, 281096, 102, 1096274, 736, 1113, 1716, 1267, 1574137, 420, 220, 32, 421643, 422, 645, 2389315, 107, 1574, 1072, 1760, 337314, 1236822, 191, 1509, 484, 281096, 3822905, 365425, 980, 228, 56219, 140547, 955726, 916, 64, 1705, 1063, 535, 1622, 562192, 252986, 365425, 267, 1058, 112438, 534082, 586, 2304987, 803, 1908, 1621, 534082, 252986, 490, 309205, 891, 1939562, 514, 1447, 1307, 1840, 265, 128, 337314, 1555, 1293041, 1517918, 708, 112438, 1464, 815, 831, 554, 416, 461, 958, 386, 1970, 1757, 1620, 970, 393534, 56219, 1707, 309205, 1000, 977, 758959, 1103, 1621, 1437, 167, 1720, 1913, 211, 1195, 1040054, 953, 112438, 449753, 1119, 1124, 140547, 1714685, 319, 393534, 449753, 1208712, 133, 462, 699, 1546, 1465, 1190, 649, 1879, 376, 625, 309205, 73, 252986, 2052000, 1546027, 145, 630, 421643, 1611, 93, 1152493, 440, 1190, 138, 3453912, 871398, 283, 96, 505972, 1454, 4609974, 1848, 334, 676, 534082, 219, 883, 321, 777, 1297, 1, 1810, 376, 301, 1323, 618411, 1799014, 1858, 1637, 1349260, 1180603, 56219, 1451, 855, 1546027, 84328, 1202, 793, 224876, 1646, 1123, 123, 1295, 759, 787, 393534, 196767, 995, 1001, 1877, 1213, 1765, 5103140, 309205, 1293041, 505972, 962, 1433589, 140547, 903, 477863, 1705, 1161, 522, 485, 1035, 1511, 65, 291, 1154, 1652, 659, 196767, 1152493, 87, 411, 899507, 1770904, 84328, 90, 365425, 1837, 955726, 610, 1448, 1714685, 1868, 804, 56219, 977, 1428, 281096, 984, 1428, 1334, 1757, 148, 5256494, 1121, 74, 1976, 140547, 1488, 1954, 1015, 830, 955, 112438, 696, 758959, 140547, 843287, 371, 1024, 920, 1985, 1434, 707, 309, 983836, 1343, 815178, 1074, 880, 386, 309205, 842, 1677, 1302, 791, 16, 702740, 615, 646520, 2192548, 1429, 77, 814, 1832, 238, 590301, 140547, 1672, 702740, 2000, 445, 1507, 766930, 2136329, 816, 168657, 1437, 554, 814, 618411, 1102, 365425, 109, 681, 820, 184, 1855233, 1542, 1096274, 1489809, 1843, 1247, 1093, 1688, 1946, 162, 1773, 196767, 1190, 553, 1669, 1174, 1303, 2304987, 168657, 590301, 327, 309205, 309205, 1656, 1904, 221, 477863, 309205, 3279773, 618411, 308, 713, 1122, 927616, 745, 899, 454, 646520, 977, 674630, 1614, 337314, 1715, 1880, 1085, 534082, 1032, 302, 578, 340, 131, 684, 1688, 1068164, 266, 1571, 1035, 309205, 452, 1477, 815178, 393534, 1405480, 373, 1124383, 196767, 1807, 1321151, 758959, 1932, 104, 534082, 421643, 1854, 1500, 1939562, 1525, 857, 1426, 426, 1124383, 815178, 1810, 371, 277, 667, 561, 1184, 56219, 920, 1918, 1747, 1602247, 84328, 609, 1701, 242, 923, 36, 871398, 590301, 590301, 407, 140547, 1026, 1403, 1321151, 730849, 1815, 562192, 168657, 416, 365425, 1257, 113, 3439115, 449753, 216, 583, 1259, 281096, 1558, 477863, 1075, 664, 681, 389, 544, 1660, 953, 1868, 505972, 1160, 281096, 1236, 1117, 1859, 365425, 536, 958, 1921, 1023, 1942, 2220658, 421643, 1234, 224876, 590301, 1347, 140547, 4556198, 590301, 624, 2023891, 177, 449753, 1280, 713, 443, 646520, 1040054, 169, 1165, 1529, 785]

general_slot = general_slot[:100]

slots_vocabulary_size = slots_vocabulary_size[:100]



bs = int(os.environ.get('bacth_size', 2))
all2all_slot = general_slot # [20] * 1000
total_slot = all2all_slot
total_all2all_slot_size = sum(all2all_slot)  # 5
embedding_dim = 2
Iters = 10

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

# 学习率
LEARNING_RATE_SPARSE = 0.005   # 0.00l5
LEARNING_RATE_DENSE = 0.0001   # 0.0001
# 常量初始化器值
CONSTANT_INIT = 1
# 合表内子表数量上限
MAX_TABLES_PER_GROUP = 50
MAX_MERGE_TABLE_CAPACITY = 350000 * 1000
MAX_MERGE_TABLE_SLOT_SIZE = 4001
# 是否启用池化模式
use_pooling_in_pull = False
# 是否运行合表测试的模式
USE_MERGED_TABLE = os.environ.get('USE_MERGED_TABLE', 'true').lower() == 'true'
# 单卡词汇表大小（假设多卡均分）
local_vocabulary_size = [ v//num_npus+1 for v in slots_vocabulary_size]

log_config(
    logger,
    bs=bs, embedding_dim=embedding_dim, Iters=Iters,
    num_npus=num_npus, uplicate_rate=uplicate_rate, conbiner=conbiner,
    use_pooling_in_pull=use_pooling_in_pull, USE_MERGED_TABLE=USE_MERGED_TABLE,
    all2all_slot_len=len(all2all_slot),
)
log_array(logger, 'all2all_slot', all2all_slot, max_show=200)
log_array(logger, 'local_vocabulary_size', local_vocabulary_size, max_show=200)


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
        with tf.variable_scope(f"merged_table_scope_{rank}_table_{self.table_id}"):
            self.global_embedding_table = tf.get_variable(
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
        #       f"key_offset:{sub_table_config.key_offset}, slot_size:{sub_table_config.slot_size}, capacity:{sub_table_config.capacity}")
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
        return tf.repeat(indices, sub_table_key_counts)

    def get_saved_context(self):
        """获取保存的中间状态（用于反向传播）"""
        return self.saved_context


class MergedTableEmbeddingEngine:
    """
    合表模式的embedding引擎

    整合了合表的前向传播和反向传播逻辑
    """

    def __init__(self, merged_table, num_npus, _key_process_func, sparse_optimizers=None, combiner=0, pooling=True):
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

        self.trace_tensors = {
            'combined_keys': combined_keys,
            'send_sizes': send_sizes,
            'recv_sizes': recv_sizes,
            'indices': indices,
            'uindex': uindex,
            'global_embeddings': global_embeddings,
            'recv_embeddings': recv_embeddings,
            'restored_embeddings': restored_embeddings,
        }
        for i, emb in enumerate(sub_table_embeddings):
            self.trace_tensors[f'sub_table_emb_{i}'] = emb
        for i, out in enumerate(sub_table_outputs):
            self.trace_tensors[f'sub_table_output_{i}'] = out

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

        init = sfps.Initializer(10, 10, 'constant')
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
       
# 输入数据
local_indices = []
local_indices_start = 0
for i in range(len(all2all_slot)):
    start = local_indices_start
    end = local_indices_start + slots_vocabulary_size[i]
    local_indices_start = end
    indices = np.random.randint(start, end, size=(bs, all2all_slot[i]),
                                dtype=np.int64)
    local_indices.append(indices)

# 生成随机二分类标签（用于AUC计算）
local_labels = np.random.randint(0, 2, size=(bs, 1)).astype(np.float32)

log_section(logger, '静态输入样本 (模块加载时生成)')
log_feed_inputs(logger, step='init', local_indices=local_indices, labels=local_labels)

def get_input(worker_rank, step, uplicate_rate=0.6):
    return local_indices, local_labels


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


if __name__ == '__main__':
    assert "worker" == os.getenv('DMLC_ROLE', "worker")
    
    from SFPS.tensorflow.TFWorker import TFWorker
    from SFPS.tensorflow.ops import _key_process
    
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
    log_section(logger, '合表分组')
    logger.info(f'  共 {len(groups)} 个大表, {len(sub_tables)} 个小表')
    for gi, group in enumerate(groups):
        caps = [sub_tables[i].capacity for i in group]
        slots = [sub_tables[i].slot_size for i in group]
        logger.info(f'  merged_table_{gi}: sub_tables={group}, capacities={caps}, slot_sizes={slots}')

    # 根据合表分组在sfps创建table
    sfps = TFWorker()
    sfps.prefetch(prefetch_step)
    create_merged_all2all_embedding_for_every_slot(sfps, groups, sub_tables)
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()
    sfps.total_embedding_count = len(sfps.table_create_infos)
    sfps.barrier()

    # 创建合表
    merged_tables, sub_table_to_merged = create_merged_embeddings(
        groups,
        sub_tables,
        embedding_dim,
        sfps.total_embedding_count,
        sparse_optimizer
    )
    print(f"   创建了 {len(merged_tables)} 个大表")
    print(f"   小表到大表的映射: {sub_table_to_merged}")
    log_section(logger, '合表创建结果')
    logger.info(f'  merged_tables 数量: {len(merged_tables)}')
    logger.info(f'  sub_table_to_merged: {sub_table_to_merged}')
    for mt in merged_tables:
        logger.info(
            f'  MergedTable id={mt.table_id}, global_capacity={mt.global_capacity}, '
            f'embedding_dim={mt.embedding_dim}, num_sub_tables={mt.get_num_sub_tables()}'
        )

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

    print("\n2. 创建input placeholders")
    label_placeholder = tf.placeholder(tf.float32, shape=[bs, 1], name='label_placeholder')
    local_input_placeholders = [
        tf.placeholder(tf.int64, shape=[bs, None], name=f'input_placeholder_{i}')
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

        print(f"   大表{eng_idx}: 包含小表 {sub_table_indices}, 使用对应的input placeholders")

        # 前向传播
        embs_before_pooling, embs = engine.lookup_and_gather(sub_inputs)

        # 按照all2all_slot顺序组装
        for idx, emb_before_pooling, emb in zip(sub_table_indices, embs_before_pooling, embs):
            all2all_embeddings_before_pooling[idx] = emb_before_pooling
            all2all_embeddings[idx] = emb


    # 拼接所有大表的输出
    all2all_embedding = tf.concat(all2all_embeddings, axis=1)
    batch_embedding = all2all_embedding
    loss, logits = dcn_model(batch_embedding, label_placeholder, is_pooling=use_pooling_in_pull)
    # 预测概率，用于AUC计算
    pred_prob = tf.sigmoid(logits, name='pred_prob')
    # AUC 指标（累计）
    auc_value, auc_update_op = tf.metrics.auc(labels=label_placeholder, predictions=pred_prob)

    # 图内 trace：各 merged engine 中间张量 + 模型输出
    trace_names = []
    trace_tensors = []
    model_trace_names = ['batch_embedding', 'logits', 'pred_prob']
    for eng_idx, engine in enumerate(merged_engines):
        if not hasattr(engine, 'trace_tensors'):
            continue
        sub_table_indices = [
            i for i, mapped_idx in enumerate(sub_table_to_merged) if mapped_idx == eng_idx
        ]
        for name, tensor in sorted(engine.trace_tensors.items()):
            if name.startswith('sub_table_'):
                local_i = int(name.rsplit('_', 1)[-1])
                global_slot = sub_table_indices[local_i]
                trace_names.append(f'merged_{eng_idx}.slot_{global_slot}.{name}')
            else:
                trace_names.append(f'merged_{eng_idx}.{name}')
            trace_tensors.append(tensor)
    trace_names.extend(model_trace_names)
    trace_tensors.extend([batch_embedding, logits, pred_prob])

    _log_trace_iters = int(os.environ.get('LOG_TRACE_ITERS', '3'))

    print(f"\n4. 模型输出shape: {all2all_embedding.shape}")
    print("\n5. 构建反向传播图")
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_DENSE)
    var_list = tf.trainable_variables()
    print(f"-----var_list: {var_list}")
    dense_vars = [v for v in var_list
                  if not any(prefix in v.name for prefix in ['merged_', 'embedding'])]
    print(f"-----dense_vars: {dense_vars}")
    dense_grads = optimizer.compute_gradients(loss, dense_vars)
    sparse_grads_for_backward = sparse_optimizer.compute_gradients(loss, all2all_embeddings_before_pooling)

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

    all_merged_update_ops = tf.group(all_merged_update_ops, name="merged_train_step_group")

    print(f"   反向传播图构建完成")
    print("\n6. 准备训练")
    sfps.train()

    # 创建session
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["mix_compile_mode"].b = True
    custom_op.parameter_map["enable_parallel_graph"].b = True
    custom_op.parameter_map["parallel_graph_thread_pool_size"].i = 64
    custom_op.parameter_map["enable_parallel_fusion"].b = True
    custom_op.parameter_map["enable_multi_stream"].b = True
    custom_op.parameter_map["stream_num"].i = 10
    config.intra_op_parallelism_threads = 64
    config.inter_op_parallelism_threads = 64
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    config.graph_options.rewrite_options.function_optimization = RewriterConfig.OFF

    with tf.Session(config=config) as sess:
        sess.run(npu_int)
        sfps.broadcast_embedding()
        sess.run(tf.global_variables_initializer())

        sess.graph.finalize()
        tf.train.write_graph(sess.graph, path, 'merged_train.pbtxt')

        log_section(logger, '训练循环开始')
        logger.info(f'  Iters={Iters}, LOG_TRACE_ITERS={_log_trace_iters} (前 N 步写详细 trace)')
        print("\n7. 开始训练循环")
        before_train = time.time()
        for i in range(Iters):
            if i == 2:
                before_train = time.time()

            read_dataset_start = time.time()
            input_local_batch, input_labels = get_input(sfps.get_group_rank(), i, uplicate_rate)
            feed_dict = {label_placeholder: input_labels}
            for placeholder, data in zip(local_input_placeholders, input_local_batch):
                feed_dict[placeholder] = data

            if i < _log_trace_iters:
                log_feed_inputs(logger, i, input_local_batch, input_labels)

            start_train = time.time()
            if i < _log_trace_iters:
                run_ops = [loss, all_merged_update_ops] + trace_tensors
                run_results = sess.run(run_ops, feed_dict=feed_dict)
                _loss = run_results[0]
                trace_values = run_results[2:]
                emb_trace_names = [n for n in trace_names if n not in model_trace_names]
                emb_trace_values = [
                    trace_values[trace_names.index(n)] for n in emb_trace_names
                ]
                log_trace_results(logger, i, emb_trace_names, emb_trace_values)
                trace_dict = dict(zip(trace_names, trace_values))
                log_model_outputs(logger, i, {
                    'loss': _loss,
                    'batch_embedding': trace_dict['batch_embedding'],
                    'logits': trace_dict['logits'],
                    'pred_prob': trace_dict['pred_prob'],
                })
                if i == 0:
                    sess.run(auc_update_op, feed_dict=feed_dict)
                    _auc = sess.run(auc_value)
                    log_array(logger, 'auc (step0)', _auc)
            else:
                _loss, _ = sess.run([loss, all_merged_update_ops], feed_dict=feed_dict)

            logger.info(
                f'[Step {i}] 完成: loss={float(_loss):.6f}, '
                f'train_cost={time.time() - start_train:.4f}s, '
                f'total_cost={time.time() - read_dataset_start:.4f}s'
            )
            print(f"iter {i}, loss={_loss:.6f}, step cost {time.time() - start_train:.4f}s")

        sfps.barrier(4)
        log_section(logger, '训练结束')
        logger.info(f'  total_train_cost={time.time() - before_train:.4f}s')
        print(f"all step train finished, cost {time.time() - before_train}")
        time.sleep(0.1)

    print(f"worker finish time = {time.time() - before_train}")
    print("\n" + "="*60)
    print("合表功能集成测试完成")
    print("="*60 + "\n")