
# hstu 训练过程

## hstu在npu上训练

### 训练启动脚本分析
[run.sh](https://gitcode.com/ww1881/RecSDK_1121/blob/develop_examples_and_tools/torch_rec_v2_examples/gr/run.sh)

``` shell
#!/bin/bash


# 加载昇腾的环境：npu工具链和运行环境 
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 如果是arm架构，比如鲲鹏服务器，需要加载openMP，做并行处理的
if [[ $(uname -m) =~ "aarch64" ]];then
    export LD_PRELOAD=/usr/lib64/libgomp.so.1
fi

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 控制device上的并发连接数  ？ 

RECSYS_DIR=$(realpath ../)
HSTU_DIR=$RECSYS_DIR/hstu
# 根据实际情况设置python引用路径
MEGATRON_DIR=$RECSYS_DIR/../../Megatron-LM/
MINDSPEED_DIR=$RECSYS_DIR/../../MindSpeed/
export PYTHONPATH=${PYTHONPATH}:${RECSYS_DIR}:${HSTU_DIR}:${MEGATRON_DIR}:${MINDSPEED_DIR}

#---------------------------------------------
# speedup
#---------------------------------------------
export TASK_QUEUE_ENABLE=2   # 打开任务队列，任务调度，异步执行效率

# cpu-binding   cpu和npu的绑定，
NPU_NUM=$(npu-smi info|grep 950PR|wc -l)
CPU_CORES=$(nproc --all)
if [ "$NPU_NUM" -eq 0 ]; then
  echo "NPU_NUM is 0, exit"
  exit 1
fi
CORES_PER_NPU=$((CPU_CORES / NPU_NUM))
CPU_AFFINITY_CONF_TMP=1
if [ "$NPU_NUM" -gt 0 ]; then
  for (( i=0; i<NPU_NUM; i++)); do
    start_core=$(( i * CORES_PER_NPU))
    end_core=$((start_core + CORES_PER_NPU -1))
    CPU_AFFINITY_CONF_TMP+=",npu${i}:${start_core}-${end_core}"   # 每个npu绑定的cpu组。示例：1,npu0:0-7，npu1:8-15  减少线程抢占，调度混乱， cpu cache miss
  done
fi
export CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF_TMP
echo "CPU_AFFINITY_CONF="$CPU_AFFINITY_CONF


#---------------------------------------------
# prof related
#---------------------------------------------
export NPU_PROFILE=0

#---------------------------------------------
# train job related
#---------------------------------------------
py_file=pretrain_gr_ranking.py    # hstu预训练主程序
config_file=movielen_ranking.gin  # 训练使用的配置文件

# 根据实际情况修改
export WORLD_SIZE=1     # 使用多少张卡进行训练
export ASCEND_RT_VISIBLE_DEVICES=0  # 训练可见的卡编号，也就是指定这几张卡可用于训练

# torchrun  pytorch官方的分布式启动器：初始化分布式环境，管理多卡训练
torchrun \
    --nproc_per_node ${WORLD_SIZE} \    # 当前机器启动x个训练进程，这些进程映射到x张卡上。本质：这个参数 = 使用的卡数 = 启动训练进程数
    --master_addr localhost \   # 分布式训练的主节点，这里是本机
    --master_port 6000 \   # 分布式通信端口：6000  如果被占用，训练失败，可以更改
    ${py_file} \
    --gin-config-file ${config_file} \      # .gin配置文件传给主程序。 一般主程序中gin.parse_config_file 来进行解析。
    2>&1 |tee temp_$(date '+%Y%m%d_%H%M%S').log   # 重定向，到标准输出。同时把输出也tee "分流" 到日志文件中

```
