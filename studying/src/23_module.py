# -*- coding: utf-8 -*-

import sys


'''
moudule
import MODULE
from MODULE import (FUN, VAR..)
说明：当使用 from package import item 这种形式的时候，对应的item既可以是包里面的子模块（子包），或者包里面定义的其他名称，比如函数，类或者变量

1、搜索路径： sys.path
   
   __init__.py : 包定义文件, 执行import 包/子包时，首先执行该文件；该文件中有个属性__all__,可以定义要导入该包中的module;

2、每个模块都有__name__属性

3、可以通过dir(模块)查看模块内定义所有名称


'''

# 增加模块的搜索路径
sys.path.append()
