worker_size=8
server_size=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/tfplugin/set_env.sh

export SFPS_KEY_PROCESS_THREADS_NUM=50

export SFPS_PROFILING_OUTPUT_PREFIX=./profiler
export HCCL_THREAD_MODE=MULTI_THREAD

# export NPU_IDS=$(npu-smi info -l | grep "NPU" | awk '{print $NF}' | tr '\n' ' ')
export NPU_IDS="0 1 2 3 4 5 6 7"
# export NPU_IDS="0 1"
first_npu="${NPU_IDS:0:1}"
echo "first_npu ${first_npu}" 

# first_npu=$(echo $NPU_IDS | awk '{print $1}')
# echo "first_npu: ${first_npu}"

export CM_CHIEF_IP=10.41.2.109  # 主节点ip
export CM_CHIEF_PORT=60205  # 主节点监听端口
export CM_CHIEF_DEVICE=$((first_npu)) # 主节点device id
export CM_WORKER_IP=10.41.2.109  # 当前节点ip
export CM_WORKER_SIZE=$worker_size  # 参与集群训练的device数量
export JOB_ID=1008632

#export ASAN_LIBS="/usr/lib/gcc/aarch64-linux-gnu/7.3.0/libasan.so:/usr/lib/gcc/aarch64-linux-gnu/7.3.0/libubsan.so"
#export LD_PRELOAD=$ASAN_LIBS
#export ASAN_OPTIONS="halt_on_error=1:log_path=./mem.err"

export PATH=$PATH:/usr/local/openmpi/bin
export HCCL_IF_BASE_PORT=31030
export HCCL_CONNECT_TIMEOUT=600

export SFPS_OP_GATHER_IN_CPU=1
export SFPS_OP_UNIQUE_IN_CPU=1
export SFPS_OP_UNIQUE_CPU_THREADS=1
export SFPS_OP_SEGMENTSUM_IN_CPU=1
export SFPS_OP_GATHER_CPU_THREADS=1
export SFPS_OP_SEGMENTSUM_CPU_THREADS=1
export SFPS_USE_NEON=0
export SFPS_SERVER_USE_NEON=0

# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

lsof -ti :10247 | xargs kill -9




mpilaunch --node-number 1 --node-id 0 --servers $server_size  --workers $worker_size --scheduler-ip 127.0.0.1 --scheduler-port 10247 --interface lo \
  --verbose 0 --default_scheduler_server \
  --env DMLC_ENABLE_RDMA:zmq --env CUDA_VISIBLE_DEVICES:1 --env BYTEPS_ENABLE_IPC:0 --env NCCL_DEBUG:INFO \
  --env SFPS_SOCKET_PATH:/tmp/zzy --env c:1  \
  python test_all2all_zm_4k_w_kp_log.py


# mpilaunch: 华为自研的 mpilaunch 工具（基于 OpenMPI 封装）启动一个分布式训练任务，常用于昇腾 NPU 上的多机多卡训练
# 该命令会启动 1 个调度器 + 8 个工作进程（相当于 8 个 rank），每个工作进程运行同一个 Python 脚本，形成分布式训练集群。
# 通过设置 OMPI_COMM_WORLD_LOCAL_RANK 等环境变量，进程可以知道自己在本地的编号，从而绑定到对应的 NPU 设备，实现 8 卡并行训练。

# --node-number 1：总节点数（物理机数），这里是单机。
# --node-id 0：当前节点编号（0 表示主节点）。
# --servers 0：参数服务器数量（此处为 0，表明使用纯 all‑reduce 或 all‑to‑all 模式，没有独立的参数服务器）。
# --workers 8：工作进程数（通常是单机上的 NPU 卡数，即 8 卡）。
# --scheduler-ip / --scheduler-port：调度器的 IP 和端口，这里用 127.0.0.1，表示调度器与训练进程在同一主机上。
# --interface lo：通信使用的网络接口，lo 是本地回环（单机模式常用）。
# --verbose 0：日志详细程度，0 表示只输出关键信息。
# --default_scheduler_server：使用内置的默认调度服务。
# --env KEY:VALUE：向每个工作进程传递环境变量。例如：
# DMLC_ENABLE_RDMA:zmq 使用 ZMQ 作为底层通信。
# CUDA_VISIBLE_DEVICES:1 将可见 GPU 设为 1（注意这里可能是示例，实际 NPU 训练会通过其他环境变量如 ASCEND_DEVICE_ID 控制）。
# BYTEPS_ENABLE_IPC:0 关闭 BytePS 的 IPC。
# NCCL_DEBUG:INFO 开启 NCCL 调试信息（尽管此处是 NPU 训练，可能兼容部分环境）。
# SFPS_SOCKET_PATH:/tmp/zzy 设置 SFPS 通信 socket 路径。
# c:1 似乎是自定义参数，可能用于控制某个特性