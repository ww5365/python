
## vllm-gr和RecSdk的区别

vllm-gr: https://github.com/vllm-project/vllm-gr
RecSdk: https://gitcode.com/Ascend/RecSDK


这两个库虽然都与“推荐系统”相关，但**定位、核心功能和所处技术栈层次完全不同**。

**vllm-gr** 是 vLLM 生态下专注于**生成式推荐模型推理与服务**的扩展框架；而 **RecSDK** 是华为昇腾生态下专注于**搜索推荐广告模型大规模训练**的 SDK。

### 总结

| 对比维度 | **vLLM-GR** | **RecSDK** |
| :--- | :--- | :--- |
| **所属生态** | vLLM 社区 | 华为昇腾生态（MindSDK） |
| **核心定位** | 生成式推荐模型的**推理与服务** | 搜索推荐广告模型的**大规模训练** |
| **关键功能** | Beam Search 加速、KV Cache 复用、目录约束、多协议服务 | 大规模稀疏表训练、特征准入/淘汰、多级存储、分布式训练 |
| **架构特点** | 扩展 vLLM 引擎，增加 Beam Session 管理 | 分层训练框架，深度集成 CANN 与 TorchRec |
| **硬件支持** | **NVIDIA GPU + 昇腾 NPU** | **昇腾 NPU**（Atlas 系列） |
| **框架依赖** | vLLM | TensorFlow、PyTorch (+ TorchRec) |

它们处于推荐系统生命周期中**训练（RecSDK）与推理（vLLM-GR）** 两个不同阶段，服务于不同的工程目标。在一个完整的基于昇腾的生成式推荐系统流水线中，它们可以形成互补：用 RecSDK 在昇腾上完成大规模稀疏特征模型的训练，再将训练好的模型（如 OneRec）交给 vLLM-GR 在昇腾 NPU 上进行高效的在线推理与服务。