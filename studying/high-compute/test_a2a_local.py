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
from memory_profiler import log_memory, log_memory_banner, estimate_merged_tables_mb
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

# general_slot = [191, 1, 1, 1, 1, 13, 1, 1, 23, 1, 1, 36, 81, 1, 1, 1, 7, 28, 1, 1, 1, 1, 3000, 1, 1, 1, 1, 1, 163, 1, 1, 65, 1, 1, 1, 1, 1, 1, 86, 1, 1, 3000, 60, 1, 1, 1, 1, 1, 1, 1, 119, 1, 1, 1, 126, 12, 1, 5, 13, 1, 1, 1, 32, 97, 1, 1, 1, 1, 130, 31, 1, 30, 1, 1, 1, 1, 146, 1, 1, 48, 1, 1, 101, 1, 97, 52, 13, 2, 8, 1, 1, 1, 1, 1, 16, 1, 1, 1, 1, 1, 1, 56, 1, 361, 157, 51, 1, 4, 1, 144, 1, 6, 1, 1, 126, 137, 65, 1, 26, 1, 1, 21, 21, 32, 1, 1, 25, 1, 1, 233, 1, 185, 1, 1, 14, 1, 1, 1, 1, 1, 1, 63, 1, 1, 130, 1, 270, 15, 1, 1, 1, 145, 53, 1, 106, 23, 85, 75, 1, 1, 36, 1, 1, 1, 1, 1, 1, 29, 2, 1, 1, 32, 1, 7, 1, 1, 1, 1, 72, 1, 1, 11, 1, 1, 15, 3000, 1, 32, 60, 1, 1, 1, 1, 1, 1, 1, 1, 30, 21, 1, 168, 1, 1, 1, 87, 23, 15, 1, 1, 1, 3000, 39, 1, 1, 73, 1, 16, 3, 23, 3, 101, 44, 1, 13, 16, 86, 1, 1, 1, 1, 1, 166, 1, 89, 23, 1, 22, 1, 1, 1, 1, 145, 56, 189, 1, 40, 1, 13, 1, 59, 1, 1, 86, 1, 13, 1, 1, 1, 63, 1, 1, 1, 17, 1, 59, 46, 1, 50, 182, 45, 67, 1, 16, 63, 1, 47, 1, 31, 1, 1, 1, 47, 1, 1, 1, 1, 1, 107, 2, 1, 1, 1, 4, 64, 1, 1, 1, 1, 32, 7, 79, 10, 1, 1, 1, 1, 42, 1, 67, 1, 3000, 3000, 1, 62, 47, 24, 1, 77, 1, 15, 1, 88, 1, 1, 1, 11, 56, 40, 122, 1, 47, 1, 2, 1, 15, 1, 72, 95, 3000, 86, 1, 1, 1, 1, 1, 1, 7, 33, 1, 1, 48, 1, 92, 1, 1, 5, 10, 1, 1, 1, 1, 61, 1, 1, 1, 41, 1, 22, 1, 1, 1, 17, 1, 43, 1, 120, 34, 1, 1, 1, 1, 35, 3000, 30, 1, 1, 66, 1, 10, 1, 193, 1, 1, 1, 80, 1, 9, 41, 6, 1, 1, 1, 1, 25, 23, 69, 1, 1, 4, 1, 91, 32, 1, 1, 1, 1, 1, 4, 1, 1, 6, 1, 1, 1, 1, 162, 59, 1, 1, 1, 1, 1, 1, 53, 8, 65, 1, 74, 1, 3000, 1, 1, 13, 1, 1, 1, 45, 1, 1, 1, 27, 60, 1, 1, 49, 1, 1, 50, 1, 61, 39, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14, 1, 2, 123, 1, 1, 1, 1, 1, 1, 1, 9, 11, 1, 1, 22, 1, 13, 76, 1, 28, 1, 17, 11, 56, 19, 1, 32, 1, 40, 2, 1, 7, 5, 1, 1, 1, 1, 1, 15, 1, 29, 1, 1, 47, 1, 7, 74, 14, 1, 1, 106, 23, 1, 12, 1, 23, 34, 1, 1, 89, 1, 1, 580, 38, 1, 1, 1, 1, 1, 30, 4, 1, 117, 1, 1, 1, 1, 1, 1, 43, 29, 1, 1, 49, 1, 1, 1, 37, 1, 1, 23, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 28, 81, 1, 1, 1, 3000, 25, 114, 3, 1, 38, 1, 1, 30, 1, 2, 93, 1, 1, 16, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 26, 146, 1, 1, 1, 1, 1, 1, 57, 1, 21, 9, 120, 1, 1, 2, 1, 1, 1, 1, 22, 1, 1, 77, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 1, 1, 1, 1, 1, 1, 152, 1, 1, 138, 1, 1, 1, 108, 142, 1, 3000, 1, 115, 31, 35, 1, 1, 75, 1, 69, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 140, 71, 1, 1, 1, 1, 97, 101, 1, 1, 1, 1, 1, 1, 18, 6, 1, 33, 1, 21, 1, 42, 11, 1, 1, 1, 1, 1, 43, 1, 95, 20, 1, 1, 151, 1, 24, 1, 1, 1, 1, 9, 1, 1, 1, 1, 85, 174, 5, 7, 1, 1, 1, 120, 187, 64, 1, 1, 1, 64, 1, 3, 76, 1, 1, 1, 1, 1, 1, 1, 110, 1, 1, 1, 1, 63, 1, 13, 34, 4, 1, 17, 1, 1, 22, 1, 1, 1, 2, 1, 69, 88, 1, 1, 1, 1, 3000, 84, 1, 1, 1, 1, 1, 1, 1, 1, 1, 46, 1, 20, 1, 1, 1, 88, 1, 50, 108, 18, 1, 38, 1, 1, 1, 3, 93, 1, 67, 1, 1, 1, 142, 57, 2, 1, 104, 1, 309, 155, 1, 1, 73, 3000, 1, 1, 1, 1, 1, 1, 1, 47, 1, 1, 1, 1, 1, 1, 1, 47, 1, 1, 11, 9, 1, 1, 1, 1, 6, 1, 97, 1, 1, 1, 1, 1, 1, 68, 69, 1, 1, 52, 1, 1, 8, 1, 1, 1, 19, 1, 1, 1, 196, 24, 41, 3000, 36, 1, 1, 1, 1, 1, 1, 44, 6, 22, 11, 1, 1, 100, 1, 1, 95, 34, 1, 1, 13, 104, 1, 3000, 1, 1, 11, 14, 26, 149, 1, 53, 28, 1, 1, 1, 1, 1, 1, 1, 18, 73, 1, 86, 1, 1, 1, 18, 1, 1, 1, 1, 36, 1, 2, 123, 1, 1, 1, 1, 1, 1, 127, 106, 55, 1, 1, 10, 1, 2, 1, 86, 13, 1, 1, 1, 1, 1, 31, 1, 19, 1, 1, 1, 1, 4, 3000, 18, 60, 2, 18, 96, 19, 1, 1, 1, 1, 1, 35, 2, 55, 141, 2, 1, 12, 1, 1, 1, 1, 83, 161, 3, 1, 1, 1, 1, 1, 1, 1]

# slots_vocabulary_size = [3056384, 460, 1223, 1173, 428, 208026, 6, 519, 368046, 1827, 937, 576072, 1296162, 352, 1214, 1784, 112014, 448056, 1554, 310, 650, 1361, 1173906, 892, 902, 798, 1893, 993, 2608327, 1195, 740, 1040130, 906, 1331, 538, 1830, 619, 536, 1376172, 1800, 152, 3163261, 960120, 1842, 73, 375, 762, 1444, 1682, 1131, 1904239, 728, 1401, 1583, 2016253, 192024, 164, 80010, 208026, 1873, 1960, 388, 512064, 1552194, 796, 896, 731, 1197, 2080261, 496062, 1523, 480060, 1066, 112, 397, 1090, 2336293, 278, 1945, 768096, 53, 1776, 1616203, 1057, 1552194, 832104, 208026, 32004, 128016, 1585, 1830, 943, 1624, 515, 256032, 1313, 1799, 1340, 1696, 1605, 1427, 896112, 1626, 5776725, 2512315, 816102, 1779, 64008, 90, 2304289, 80, 96011, 1961, 1134, 2016253, 2192275, 1040130, 883, 416052, 1330, 387, 336042, 336042, 512064, 1670, 1230, 400049, 501, 359, 3728468, 148, 2960371, 1973, 546, 224027, 1885, 145, 1800, 120, 818, 1548, 1008127, 724, 1881, 2080261, 568, 4320542, 240030, 1592, 1882, 1162, 2320291, 848106, 954, 1696213, 368046, 1360171, 1200150, 301, 3, 576072, 1912, 580, 992, 1903, 1754, 838, 464058, 32004, 331, 1349, 512064, 702, 112014, 1019, 367, 799, 817, 1152144, 427, 914, 176021, 1711, 1860, 240030, 1971894, 892, 512064, 960120, 1263, 1699, 80, 3, 1889, 918, 185, 56, 480060, 336042, 1236, 2688337, 1254, 625, 1882, 1392175, 368046, 240030, 1485, 1852, 294, 3028201, 624078, 915, 1610, 1168146, 865, 256032, 48005, 368046, 48005, 1616203, 704088, 557, 208026, 256032, 1376172, 569, 829, 716, 1251, 1883, 2656333, 718, 1424178, 368046, 1255, 352043, 1002, 499, 666, 40, 2320291, 896112, 3024380, 1690, 640080, 192, 208026, 898, 944118, 1524, 1777, 1376172, 306, 208026, 131, 1963, 742, 1008127, 456, 1222, 1768, 272033, 315, 944118, 736092, 1956, 800100, 2912365, 720090, 1072134, 1864, 256032, 1008127, 233, 752094, 1039, 496062, 559, 536, 137, 752094, 1600, 309, 778, 288, 393, 1712214, 32004, 1524, 983, 1917, 64008, 1024128, 1692, 397, 1816, 1403, 512064, 112014, 1264159, 160020, 562, 1301, 292, 1386, 672084, 44, 1072134, 1033, 695750, 3005608, 454, 992124, 752094, 384048, 1178, 1232154, 830, 240030, 1259, 1408176, 1577, 1481, 904, 176021, 896112, 640080, 1952245, 1888, 752094, 1759, 32004, 895, 240030, 1116, 1152144, 1520191, 1421617, 1376172, 1438, 1452, 1310, 373, 688, 1359, 112014, 528065, 1039, 366, 768096, 1236, 1472184, 1972, 600, 80010, 160020, 781, 1697, 986, 939, 976122, 520, 610, 1157, 656082, 1655, 352043, 1051, 292, 1153, 272033, 1450, 688086, 1598, 1920241, 544068, 192, 595, 849, 314, 560070, 3288649, 480060, 612, 1058, 1056132, 252, 160020, 611, 3088387, 1368, 346, 1430, 1280160, 1526, 144017, 656082, 96011, 1270, 717, 1224, 963, 400049, 368046, 1104138, 116, 1797, 64008, 1163, 1456182, 512064, 346, 1125, 720, 854, 1604, 64008, 1090, 1001, 96011, 486, 441, 1617, 1225, 2592325, 944118, 1373, 288, 487, 1555, 1926, 1231, 848106, 128016, 1040130, 1019, 1184149, 642, 378370, 213, 700, 208026, 288, 24, 1600, 720090, 971, 1083, 1789, 432054, 960120, 306, 704, 784098, 1184, 429, 800100, 1460, 976122, 624078, 953, 1856, 344, 238, 1345, 1026, 233, 322, 1208, 371, 356, 224027, 1217, 32004, 1968247, 445, 280, 79, 232, 1283, 1862, 1567, 144017, 176021, 443, 913, 352043, 4, 208026, 1216152, 1172, 448056, 53, 272033, 176021, 896112, 304038, 827, 512064, 1250, 640080, 32004, 1622, 112014, 80010, 524, 1135, 1003, 1834, 1443, 240030, 881, 464058, 1271, 1681, 752094, 430, 112014, 1184149, 224027, 818, 202, 1696213, 368046, 944, 192024, 1612, 368046, 544068, 234, 1476, 1424178, 1972, 112, 9281165, 608076, 1693, 1386, 1115, 1836, 7, 480060, 64008, 95, 1872235, 1070, 221, 1946, 360, 1527, 1026, 688086, 464058, 1911, 1890, 784098, 1893, 1422, 1178, 592074, 122, 1166, 368046, 441, 192, 488, 968, 1031, 895, 1096, 1634, 615, 1710, 1686, 285, 448056, 1296162, 710, 1773, 1264, 2800204, 400049, 1824229, 48005, 1383, 608076, 726, 1960, 480060, 355, 32004, 1488187, 400, 1876, 256032, 1455, 378, 722, 1954, 132, 771, 1809, 48005, 265, 844, 416052, 2336293, 694, 530, 1859, 786, 1434, 1554, 912114, 493, 336042, 144017, 1920241, 867, 1632, 32004, 1551, 560, 746, 356, 352043, 1219, 200, 1232154, 871, 902, 1786, 1040, 1679, 1546, 1889, 1408, 1326, 1877, 112014, 629, 874, 138, 707, 1530, 1595, 2432305, 186, 995, 2208277, 92, 1345, 1222, 1728217, 2272285, 1405, 1549616, 1476, 1840231, 496062, 560070, 1814, 240, 1200150, 1782, 1104138, 203, 1911, 1850, 1654, 472, 419, 343, 405, 815, 608, 1282, 1981, 32004, 1320, 862, 1353, 401, 223, 48005, 585, 2240281, 1136143, 1715, 1196, 472, 1864, 1552194, 1616203, 9, 901, 1130, 598, 1978, 1363, 288036, 96011, 732, 528065, 1690, 336042, 215, 672084, 176021, 1018, 30, 186, 1567, 1022, 688086, 49, 1520191, 320040, 1099, 1487, 2416303, 1494, 384048, 1423, 680, 1629, 940, 144017, 55, 969, 827, 786, 1360171, 2784349, 80010, 112014, 941, 163, 446, 1920241, 2992375, 1024128, 1825, 856, 1721, 1024128, 704, 48005, 1216152, 1200, 911, 1396, 720, 1197, 618, 635, 1760220, 961, 992, 377, 1317, 1008127, 1285, 208026, 544068, 64008, 1982, 272033, 1711, 360, 352043, 49, 1551, 241, 32004, 1577, 1104138, 1408176, 635, 864, 768, 423, 3799268, 1344168, 884, 947, 887, 1180, 1711, 48, 1932, 1745, 214, 736092, 1856, 320040, 1326, 1309, 1756, 1408176, 85, 800100, 1728217, 288036, 1099, 608076, 873, 1867, 1794, 48005, 1488187, 1741, 1072134, 788, 191, 462, 2272285, 912114, 32004, 1997, 1664209, 235, 4944621, 2480311, 1894, 1549, 1168146, 473744, 1373, 1300, 1300, 1126, 157, 1727, 1414, 752094, 390, 542, 347, 1365, 980, 58, 1653, 752094, 769, 603, 176021, 144017, 266, 1608, 69, 1415, 96011, 1965, 1552194, 1150, 1714, 1324, 1750, 482, 520, 1088136, 1104138, 850, 380, 832104, 1691, 1569, 128016, 896, 833, 317, 304038, 803, 1671, 873, 3136393, 384048, 656082, 3283059, 576072, 168, 195, 1925, 1690, 1385, 360, 704088, 96011, 352043, 176021, 521, 735, 1600201, 1900, 540, 1520191, 544068, 214, 1825, 208026, 1664209, 156, 2050433, 513, 1659, 176021, 224027, 416052, 2384299, 696, 848106, 448056, 1753, 845, 708, 732, 1045, 420, 1430, 288036, 1168146, 1310, 1376172, 1185, 604, 1006, 288036, 697, 42, 517, 154, 576072, 52, 32004, 1968247, 1551, 1086, 1371, 446, 571, 862, 2032255, 1696213, 880110, 1078, 148, 160020, 822, 32004, 453, 1376172, 208026, 710, 529, 1509, 1156, 1211, 496062, 926, 304038, 746, 318, 501, 1812, 64008, 3815728, 288036, 960120, 32004, 288036, 1536193, 304038, 28, 1058, 202, 225, 206, 560070, 32004, 880110, 2256283, 32004, 711, 192024, 1075, 217, 1760, 1664, 1328166, 2576323, 48005, 1925, 403, 504, 1684, 842, 1713, 1970]


general_slot = general_slot[:100]

slots_vocabulary_size = slots_vocabulary_size[:100]
bs = int(os.environ.get('bacth_size', 10))
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
MAX_MERGE_TABLE_SLOT_SIZE = 4001 #4001
# 是否启用池化模式
use_pooling_in_pull = False
# 是否运行合表测试的模式
USE_MERGED_TABLE = os.environ.get('USE_MERGED_TABLE', 'true').lower() == 'true'

# 单卡词汇表大小（假设多卡均分）
local_vocabulary_size = [ v//num_npus+1 for v in slots_vocabulary_size]

print(f"Hyperparameters: bs={bs}, embedding_dim={embedding_dim}, Iters={Iters}, "
            f"local_vocabulary_size={local_vocabulary_size}, all2all_slot={all2all_slot}, "
            f"num_npus={num_npus}, uplicate_rate={uplicate_rate}, conbiner={conbiner}")

log_config(
    logger,
    bs=bs, embedding_dim=embedding_dim, Iters=Iters,
    num_npus=num_npus, uplicate_rate=uplicate_rate, conbiner=conbiner,
    use_pooling_in_pull=use_pooling_in_pull, USE_MERGED_TABLE=USE_MERGED_TABLE,
    all2all_slot_len=len(all2all_slot),
    local_vocabulary_size=local_vocabulary_size,
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

        数据处理示例：
        input_ids_list = [
        slot_0: (10,1) → [1721, 1496, 156, 135, 22, 1496, 461, 1721, 50, 643],
        slot_1: (10,1) → [1950, 1997, 2001, 1988, 1994, ...],
        slot_4: (10,1) → [2483532, 2482255, ...],
        ... 共 50 个小表
        ]
        sub_table_key_counts = [10, 10, 10, ..., 10]   # 50 个 10
        combined_global_keys: shape=(500,)   # 50 × 10

        [   1721,1496,156,135,22,1496,461,1721,50,643,     ← slot_0  [0:10)
            1950,1997,2001,1988,1994,1995,1960,1988,2007,1997,  ← slot_1  [10:20)
            2483532,2482255,...,2483248,                    ← slot_4  [20:30)
            ...共 50 段，每段 10 个]

        """
        # Step 1: 将每个小表的key转换为global_key
        global_keys_list = []
        sub_table_key_counts = []

        for i, input_ids in enumerate(input_ids_list):
            flat_input_ids = tf.reshape(input_ids, [-1])   #  input_ids tf.Tensor类型，-1 表示自动推断维度的大小，展平为一维，(10,1) -> (10,)
            global_keys_list.append(flat_input_ids)

            # 记录该小表的key数量
            num_keys = tf.shape(flat_input_ids)[0]
            sub_table_key_counts.append(num_keys)

        # Step 2: 拼接所有小表的global_keys
        # merged_0.combined_keys: shape=(500,)
        combined_global_keys = tf.concat(global_keys_list, axis=0)  # global_keys_list是list[tf.Tensor] 类型，拼接所有小表的global_keys [（10，）, (10,)...] -> (500,)

        # Step 3: 执行_key_process (使用合并后的大表参数)
        send_sizes, recv_sizes, indices, uindex, offset_count = self._key_process(
            combined_global_keys,   # shape: (500,)
            self.merged_table.table_id,  # 大表id
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
        new_capacity = current_capacity + table.capacity  # local_vocabulary_size：这是单机词汇量，除以了8卡后的结果
        new_slot_size = current_slot_size + table.slot_size

        # 检查是否需要开新组：分组策略 1：容量超过最大容量，2：小表数量超过最大小表数量，3：slot数超过最大slot配置
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
        merged_slot_size = sum(sub_tables[idx].slot_size for idx in group)  # 合表功能后，按照这个组中所有小表的slot大小，创建一个大的slot
        # 创建sfps中的table， 参数：
        # embedding_type.all2all: 表示使用all2all通信方式
        # merged_vocabulary_size: 大表总容量
        # embedding_dim: embedding维度
        # bs: batch size
        # [merged_slot_size]: slot大小
        # opt: optimizer
        # init: 初始化器
        # sfps.c_lib.key_type.int64: 键类型
        # sfps.c_lib.pooling_type.sum: 池化类型
        # sfps.c_lib.hash_type.hash: 哈希类型
        # comm_policy: 通信策略
        # feature_policy: 特征策略
        # None: 填充参数

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
    
    # 计算需要的总数据量
    total_size = bs * all2all_slot[i]
    
    if slots_vocabulary_size[i] < 8:
        # 当词汇量小于8时的处理
        new_data_size = max(1, int(total_size * 0.1))
        # 确保新数据量不超过可用数据量
        new_data_size = min(new_data_size, end - start)
        
        # 生成新数据（不重复）
        all_possible = np.arange(start, end)
        new_indices = np.random.choice(all_possible, size=new_data_size, replace=False)
        
        # 从新数据中重复选择来填充剩余90%
        repeat_indices = np.random.choice(new_indices, size=total_size - new_data_size, replace=True)
        indices = np.concatenate([new_indices, repeat_indices])
        
        # 随机打乱
        np.random.shuffle(indices)
        indices = indices.reshape(bs, all2all_slot[i])
        local_indices.append(indices)
        continue
    
    # 对于词汇量>=8的情况，按8卡均匀分布
    indices_count = total_size
    count_per_mod8 = indices_count // 8
    remaining = indices_count % 8
    
    indices_list = []
    
    for mod8 in range(8):
        current_count = count_per_mod8
        if mod8 < remaining:
            current_count += 1
        
        # 获取当前模8的所有可能值
        possible_values = np.arange(start + mod8, end, 8)
        
        if len(possible_values) == 0:
            # 如果这个模8类没有值，使用相邻模8类的值
            possible_values = np.arange(start, end, 1)
        
        # 计算新数据的数量（10%）
        new_data_count = max(1, int(current_count * 0.1))
        new_data_count = min(new_data_count, len(possible_values))
        
        # 生成新的唯一数据
        new_indices = np.random.choice(possible_values, size=new_data_count, replace=False)
        
        # 生成重复数据（90%）
        repeat_count = current_count - new_data_count
        if repeat_count > 0:
            repeat_indices = np.random.choice(new_indices, size=repeat_count, replace=True)
            mod8_indices = np.concatenate([new_indices, repeat_indices])
        else:
            mod8_indices = new_indices
        
        # 随机打乱这个模8类内的顺序
        np.random.shuffle(mod8_indices)
        indices_list.append(mod8_indices)
    
    # 合并所有模8类的数据
    indices = np.concatenate(indices_list)
    # 全局打乱以确保8卡数据混合
    np.random.shuffle(indices)
    indices = indices.reshape(bs, all2all_slot[i])
    local_indices.append(indices)

# 生成随机二分类标签（用于AUC计算）
local_labels = np.random.randint(0, 2, size=(bs, 1)).astype(np.float32)

log_section(logger, '静态输入样本 (模块加载时生成)')
log_feed_inputs(logger, step='init', local_indices=local_indices, labels=local_labels)

"""
生成的示例数据：按照  slots_vocabulary_size   slot0值域：[0, 1948）...  每个slot的key是不重复的。
  input_slot_0: shape=(10, 1), dtype=int64, values=[1721, 1496, 156, 135, 22, 1496, 461, 1721, 50, 643]
  input_slot_1: shape=(10, 1), dtype=int64, values=[1950, 1997, 2001, 1988, 1994, 1995, 1960, 1988, 2007, 1997]
  input_slot_2: shape=(10, 4000), dtype=int64, values=[1386399, 834833, 128597, 1079602, 182810, 812460, 523260, 280942, 243546, 1228983, 351401, 174563, 196711, 1311190, 879133, 321763, 1083727, 1339664, 624521, 1125966, 1385289, 501575, 1538759, 9399, 1354699, 1137697, 528061, 144221, 805458, 967226, 1449481, 1474958, 460781, 264890, 1323399, 1421647, 123792, 1536494, 1016047, 286844, 1113084, 1184881, 824266, 137454, 426966, 1546984, 1201056, 44259, 558490, 274088, 1341207, 361170, 897645, 833243, 1095914, 991684, 1310468, 1517624, 534139, 637290, 500673, 1541180, 1350390, 640886, ... (+39936 more)]
  input_slot_3: shape=(10, 32), dtype=int64, values=[1964244, 2439021, 2004650, 1749036, 1869881, 2389167, 2179508, 1937598, 2265442, 2265442, 1718211, 1749036, 1869881, 1989385, 2101929, 2130407, 2011237, 2130407, 2389167, 2367718, 2004650, 2304647, 1749036, 2265442, 1937598, 2179508, 1658168, 2130407, 2068699, 2389167, 2004650, 2004650, 1658168, 1937598, 1937598, 1658168, 1964244, 1937598, 1692995, 2474904, 2474904, 1718211, 2367718, 1869881, 2068699, 2136502, 1658168, 2474904, 2439021, 2389167, 2004650, 1718211, 2004650, 2130407, 2395476, 2389167, 2136502, 1937598, 1718211, 1757192, 2474904, 2179508, 2395476, 2263946, ... (+256 more)]
  input_slot_4: shape=(10, 1), dtype=int64, values=[2483532, 2482255, 2482494, 2482805, 2482255, 2483065, 2482494, 2483706, 2483099, 2483248]
  ... 其余 95 个 slot 仅记录 shape
  input_slot_5: shape=(10, 1), dtype=int64
  input_slot_6: shape=(10, 1), dtype=int64

"""

# 检查数据重复率
for i, indices in enumerate(local_indices):
    unique_indices = np.unique(indices)
    total_indices = indices.size
    repeat_rate = 1 - (unique_indices.size / total_indices)
    print(f"*****************table-{i}, repeat rate: {repeat_rate:.2%}")
    if repeat_rate < 0.9:
        print(f"Warning: table {i}, slot_size {general_slot[i]}, vocab_size {slots_vocabulary_size[i]}, low repeat rate: {repeat_rate:.2%}")

def get_input(worker_rank, step, uplicate_rate=0.6):
    """
    数据生成逻辑：
        1. 全局 ID 空间：所有 slot 的 key 共享一个连续的 ID 区间，
           第 i 个 slot 的 key 取值范围为 [vocab_start, vocab_start + slots_vocabulary_size[i])，其中 vocab_start 为前序 slot 的容量累加。
           这步保证了：不同slot中的数据key是不重复的。

        2. 按词汇量分两种策略：
            - vocab < 8：从整个词汇范围中随机抽取 10% 的无重复 key 作为“新 key”，再从这批新 key 中有放回地重复采样，填满剩余的 90%，最后打乱。
            - vocab ≥ 8：将总 key 数量按 mod8 均匀分到 8 个“伪卡类别”中（余数依次补足），每个类别内部同样抽取 10% 无重复 key 并重复采样至该类别总量，最后拼接所有类别的 key 并全局打乱。这样既能模拟多卡均匀分布，又保证了每类内部的重复率。
        3. 拼接并 reshape：最终将每个 slot 的 key 组 reshape 为 [bs, slot_size]。
    
    重复率计算逻辑：
        1. 计算每个 slot 的 key 数量：slots_vocabulary_size

    """
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
    # sorted_indices = tf.argsort(predictions, axis=0, direction='DESCENDING')
    sorted_indices = tf.argsort(-predictions, axis=0)
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


if __name__ == '__main__':
    assert "worker" == os.getenv('DMLC_ROLE', "worker")
    
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
    """
    local_vocabulary_size: 100 长度 
    [4,9,197592,112439,210,88,212,238,189740,33,330288,45679,351370,40,28110,87843,66761,102,84329,56220,31,126494,240,168,126494,4,134,445894,13]
    all2all_slot: 100
    [1,1,4000,...]

    输出：
    共 4 个大表, 100 个小表
    merged_table_0: sub_tables=[0, 1, 4, 5, 6, 7, 9, 13, 17, 20, 22, 23, 25, 26, 28, 29, 30, 31, 33, 35, 36, 40, 41, 42, 43, 46, 47, 48, 49, 50, 53, 54, 56, 57, 59, 60, 61, 62, 63, 66, 69, 70, 73, 77, 79, 84, 85, 87, 89, 91], capacities=[244, 9, 210, 88, 212, 238, 33, 40, 102, 31, 240, 168, 4, 134, 13, 126, 196, 199, 69, 218, 131, 93, 214, 197, 179, 158, 238, 77, 115, 238, 245, 26, 153, 93, 239, 184, 96, 74, 75, 181, 9, 151, 26, 213, 225, 122, 166, 20, 202, 124], slot_sizes=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    merged_table_1: sub_tables=[3, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21, 24, 32, 34, 37, 38, 39, 44, 45, 51, 52, 55, 58, 64, 65, 67, 68, 71, 72, 74, 75, 76, 78, 80, 81, 82, 83, 86, 88, 90, 92, 93, 94, 95, 96, 97, 98, 99], capacities=[112439, 189740, 330288, 45679, 351370, 28110, 87843, 66761, 84329, 56220, 126494, 126494, 63247, 66761, 7028, 7028, 133521, 21083, 165144, 87843, 154603, 91357, 439213, 87843, 200281, 21083, 105411, 42165, 14055, 108925, 154603, 151090, 122980, 17569, 38651, 137035, 165144, 56220, 256501, 17569, 119466, 109, 10542, 9, 91357, 171, 165, 51], slot_sizes=[32, 54, 94, 13, 100, 8, 25, 19, 24, 16, 36, 36, 18, 19, 2, 2, 38, 6, 47, 25, 44, 26, 125, 25, 57, 6, 30, 12, 4, 31, 44, 43, 35, 5, 11, 39, 47, 16, 73, 5, 34, 1, 3, 1, 26, 1, 1, 1]
    merged_table_2: sub_tables=[2], capacities=[197592], slot_sizes=[4000]
    merged_table_3: sub_tables=[27], capacities=[445894], slot_sizes=[4000]

    """
    log_section(logger, '合表分组')
    logger.info(f'  共 {len(groups)} 个大表, {len(sub_tables)} 个小表')
    for gi, group in enumerate(groups):
        caps = [sub_tables[i].capacity for i in group]
        slots = [sub_tables[i].slot_size for i in group]
        logger.info(f'  merged_table_{gi}: sub_tables={group}, capacities={caps}, slot_sizes={slots}')
		
    log_memory('01_after_merge_groups', extra={'num_groups': len(groups), 'num_sub_tables': len(sub_tables)})

    # 根据合表分组在sfps创建table
    sfps = TFWorker()
    sfps.prefetch(prefetch_step)
    create_merged_all2all_embedding_for_every_slot(sfps, groups, sub_tables)
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()
    sfps.total_embedding_count = len(sfps.table_create_infos)
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
    
    '''
    
    merged_tables 数量: 4
    sub_table_to_merged: [0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    MergedTable id=0, global_capacity=6838, embedding_dim=2, num_sub_tables=50
    MergedTable id=1, global_capacity=4761590, embedding_dim=2, num_sub_tables=48
    MergedTable id=2, global_capacity=197592, embedding_dim=2, num_sub_tables=1
    MergedTable id=3, global_capacity=445894, embedding_dim=2, num_sub_tables=1

    sub_table_to_merged: 小表(slot)属于哪个大表，索引标识的是小表slot

    '''

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

        # sub_table_indices： 大表0: 包含小表 [0, 1, 4, 5, 6, 7, 9, 13, 17, 20, 22, 23, 25, 26, 28, 29, 30, 31, 33, 35, 36, 40, 41, 42, 43, 46, 47, 48, 49, 50, 53, 54, 56, 57, 59, 60, 61, 62, 63, 66, 69, 70, 73, 77, 79, 84, 85, 87, 89, 91], 使用对应的input placeholders
        # sub_inputs: [slot0[1721, 1496, 156, 135, 22, 1496, 461, 1721, 50, 643], slot1shape(10,1), slot4shape(10,1)...] 每个分组输入 
        print(f"   大表{eng_idx}: 包含小表 {sub_table_indices}, 使用对应的input placeholders")

        # 前向传播， 对于大表0 sub_inputs: 50 * （10,1）
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
    # auc_value, auc_update_op = tf.metrics.auc(labels=label_placeholder, predictions=pred_prob)
    
    # auc_value = batch_auc(label_placeholder, pred_prob)
    batch_emb_mean = tf.reduce_mean(batch_embedding)
    batch_emb_std = tf.math.reduce_std(batch_embedding)
    batch_emb_shape = tf.shape(batch_embedding)

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

    _log_trace_iters = int(os.environ.get('LOG_TRACE_ITERS', '1'))

    print(f"\n4. 模型输出shape: {all2all_embedding.shape}")
    log_memory('05_after_build_forward_graph', merged_tables=merged_tables, embedding_dim=embedding_dim)

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
    log_memory('06_after_build_full_graph_backward', merged_tables=merged_tables, embedding_dim=embedding_dim,
               extra={'use_pooling': use_pooling_in_pull, 'bs': bs, 'total_keys_per_step': bs * sum(all2all_slot)})

    print("\n6. 准备训练")
    sfps.train()
    log_memory('07_after_sfps_train')

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
        log_memory('08_session_created')
        sess.run(npu_int)
        log_memory('09_after_sess_run_npu_int')
        sfps.broadcast_embedding()
        sess.run(tf.global_variables_initializer())
        log_memory('10_after_global_variables_initializer',
                   merged_tables=merged_tables, embedding_dim=embedding_dim,
                   extra={'note': 'embedding tables materialized on NPU here'})

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
                
                _loss = sess.run(loss, feed_dict=feed_dict)
                log_memory('iter0_after_forward_loss_only',
                           extra={'loss': float(_loss)})
                run_ops = [all_merged_update_ops] + trace_tensors
                run_results = sess.run(run_ops, feed_dict=feed_dict)
                log_memory('iter0_after_backward_sparse_dense_update')
                trace_values = run_results[1:]
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
                # if i == 0:
                #     # sess.run(auc_update_op, feed_dict=feed_dict)
                #     _auc = sess.run(auc_value)
                #     log_array(logger, 'auc (step0)', _auc)
            else:
                _loss, _ = sess.run([loss, all_merged_update_ops], feed_dict=feed_dict)
                log_memory('iter%d_after_full_train_step' % i)

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
    # print("\n" + "="*60)
    print("合表功能集成测试完成")
    print("="*60 + "\n")