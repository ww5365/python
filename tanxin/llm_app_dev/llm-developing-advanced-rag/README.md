# advanced-rag

## 项目介绍
Q：我们这节课要讲什么？

A：上节课讲了`进阶RAG的一些方法理论`，这节课讲`如何代码实现进阶RAG`

Q：本节课怎么讲？

A：围绕一个任务：分别用`简单RAG`和`进阶RAG`去实现`基于PDF文档的大模型问答`，使用`Ragas评估框架`去评估RAG的性能。

一步步实现：

- 0.ipynb：一篇新闻文章，有一个（问题，答案）对 - 简单RAG - Ragas评估
- 1.ipynb：一份PDF文档，有三个（问题，答案）对 - 简单RAG - Ragas评估
- 2.ipynb：一份PDF文档，有三个（问题，答案）对 - 进阶RAG（检索器优化） - Ragas评估
  - 仅embedding retriever -> bm25 retriever + embedding retriever = ensemble retriever
- 3.ipynb：一份PDF文档，有三个（问题，答案）对 - 进阶RAG（优化后的检索器 + 生成器优化） - Ragas评估
  - 3.1.ipynb：优化后的检索器 + 基于LLMChainExtractor的文本压缩器 = 上下文压缩
  - 3.2.ipynb：优化后的检索器 + 基于BGE Reranker的自定义文本压缩器 = 上下文压缩

涉及到的知识点：

- 检索器优化
    - [BGE Embedding](https://huggingface.co/BAAI/bge-large-zh-v1.5)
    - [BM25 Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble)
    - [Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble)
- 生成器优化
    - [Contextual compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression)
        - [LLMChainExtractor](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression#adding-contextual-compression-with-an-llmchainextractor)
        - [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-large)
- 评估
  - [Ragas评估框架](https://docs.ragas.io/en/stable/)

## 运行

1、创建虚拟环境并安装依赖
- conda create -n py38_AdvancedRAG python=3.8
- conda activate py38_AdvancedRAG
- pip install -r requirements.txt


**注意1：**

如果遇到这种错误：`ModuleNotFoundError: No module named 'pwd'`

解决方案：
- pip uninstall langchain-community
- pip install langchain-community==0.0.19

**注意2：**

在安装`sentence-transformers`包时，会自动安装 cpu 版本的 torch。

如果你电脑有 GPU ，想要安装 gpu 版本的 torch。

可以运行`tests`文件下的`test_gpu.py`查看本机环境并按照注释安装。


2、配置环境变量
   - 打开`.env.example`文件
   - 填写完整该文件中的`OPENAI_API_KEY`、`HTTP_PROXY`、`HTTPS_PROXY`、`HUGGING_FACE_ACCESS_TOKEN`四个环境变量
     - `HUGGING_FACE_ACCESS_TOKEN`获取方式：https://huggingface.co/settings/tokens
   - 把`.env.example`文件重命名为`.env`


3、运行 Jupyter Lab
   - 命令行或终端运行`jupyter lab`
     - `(py38_AdvancedRAG) D:\Work\GreedyAI\GiteeProjects\llmdeveloping-advanced-rag>jupyter lab `
     - 浏览器会自动弹出网页：`http://localhost:8888/lab`


4、下载 embedding model
- 运行`test_bge-large-zh-v1.5.ipynb`下载并测试`BAAI/bge-large-zh-v1.5`模型
- 模型 Hugging Face 地址：[https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)


5、下载 reranker model
- 运行`test_bge-reranker-large.ipynb`下载并测试`BAAI/bge-reranker-large`模型
- 模型 Hugging Face 地址：[https://huggingface.co/BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)

6、运行根目录下几个ipynb文件：0、1、2、3.1、3.2
