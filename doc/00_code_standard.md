
[TOC]

# python 代码规范


## 1. 排版

### 1.1 采用4个空格缩进风格

### 1.4 文件使用utf-8 编码
``` python
#coding: utf-8
```
### 1.6 相对独立的程序块之间、变量说明之后必须加空行


``` python
if len(deviceName) < _MAX_NAME_LEN:
...
//这里有一行空行
writer = LogWriter()
```


### 规则1.8 在两个以上的关键字、变量、常量进行对等操作时，它们之间的操作符前后要加空格 

``` python
a = b + c
a += 2
if current_time >= MAX_TIME_VALUE:
    a = b * c
    a = c ** b
    x = x*2 - 1   # "*"、"**"等作为操作符时，前后可以加空格； 特殊情况建议：但若和更低优先级的操作符同时使用并且不涉及括号，则建议前后不加空格

```

### 规则1.10 加载模块必须分开每个模块占一行
### 规则1.11 导入部分(imports)置于模块注释和文档字符串之后，模块全局变量和常量声明之前

``` python

import sys
import os

from sys import stdin, stdout   # 特殊情况：虽然一行只能加载一个模块，但同一个模块内的多个符号可以在同一行加载

# 导入(import)库时，按照标准库、第三方关联库、本地特定的库/程序顺序导入，并在这几组导入语句之间增加一个空行
import sys
import os

from oslo_config import cfg
from oslo_log import log as logging

from cinder import context
from cinder import db
```

## 2 注释


### 类、接口和函数

#### 规则2.1 类和接口的文档字符串写在类声明(classClassName:)所在行的下一行，并向后缩进4个空格

#### 规则2.2 公共函数的文档字符串写在函数声明(defFunctionName(self):)所在行的下一行，并向后缩进4个空格

####  规则2.3 公共属性的注释写在属性声明的上方，与声明保持同样的缩进。行内注释应以#和一个空格作为开始，与后面的文字注释以一个空格隔开

``` python

class TreeError(libxmlError):
    """
    功能描述：
    接口清单：

    说明： 类和接口的文档字符串的内容可选择包括（但不限于）功能描述，接口清单等。功能描述除了描述类或接口功能外，还要写明与其他类或接口之间的关系；接口清单列出该类或接口的接口方法的描述；
    """

def load_batch(fpath):
    """
    功能描述：
    参数：
    返回值：
    异常描述：
    """
    
    # Compensate for border
    x = x + 1
```

### 格式

#### 规则2.4 模块文档字符串写在文件的顶部，导入(import)部分之前的位置，不需要缩进

``` python
"""
功 能：XXX类，该类主要涉及XXX功能
版权信息：** 公司，版本所有(C) 2010-2019
"""
```

```






## 参考
