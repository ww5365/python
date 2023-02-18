
[TOC]

# python 环境搭建


## pip 源

windows 配置pip源 
C:\Users\用户名\pip\pip.ini

```
[global]
trusted-host=cmc-cd-mirror.rnd.huawei.com
index-url=http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/
```


## Anaconda下载


anaconda [下载地址](https://www.anaconda.com/products/distribution#download-section)
Anaconda3-2022.10-Windows-x86_64.exe 里面是3.9 Python版本

环境变量手动配置
！[配置anaconda环境变量](./assets/01_anaconda_environment_var.png)


conda create -n pytorch python=3.9

C:\Users\YuZhe\.conda\envs\pytorch

activate pytorch


C:\Users\YuZhe\.condarc

anaconda所谓的创建虚拟环境其实就是安装了一个真实的python环境, 只不过我们可以通过activate,conda等命令去随意的切换我们当前的python环境, 用不同版本的解释器和不同的包环境去运行python脚本



show_channel_urls: true
channels:
  - https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/



C:\Users\YuZhe\pip.ini    pip的源，清华的

[global]
	index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
    trusted-host=mirrors.aliyun.com





补充conda命令使用

要查看当前环境中所有安装了的包可以用

conda list

导入导出环境
如果想要导出当前环境的包信息可以用

conda env export > environment.yaml

将包信息存入yaml文件中.

当需要重新创建一个相同的虚拟环境时可以用

conda env create -f environment.yaml

其实命令很简单对不对, 我把一些常用的在下面给出来, 相信自己多打两次就能记住

activate // 切换到base环境

activate learn // 切换到learn环境

conda create -n learn python=3 // 创建一个名为learn的环境并指定python版本为3(的最新版本)

conda env list // 列出conda管理的所有环境

conda list // 列出当前环境的所有包

conda install requests 安装requests包

conda remove requests 卸载requets包

conda remove -n learn –all // 删除learn环境及下属所有包

conda update requests 更新requests包

conda env export > environment.yaml // 导出当前环境的包信息

conda env create -f environment.yaml // 用配置文件创建新的虚拟环境


在pytorch虚拟环境中安装pytorch:
conda install pytorch torchvision


在pytorch的虚拟环境中，启用jupyter, 并验证pytorch是否生效

conda install nb_conda










