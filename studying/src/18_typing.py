# -*- encoding: utf-8 -*-

from typing import List, Tuple, NewType

'''
本文主要介绍：python 类型注解库的使用

from typing import 

List
Dict
Tuple
NewType
Callable
Mapping
Sequence
TypeVar
Generic
Iterable
Union
Any

typing模块的作用：

类型检查: 防止运行时出现参数和返回值类型不符合
可读性：  作为开发文档附加说明，方便使用者调用传入和返回类型
效率：    该模块加入之后并不会影响程序的运行，不会报正式的错误，只有提醒

'''

# quick start


def greet_name(name: str) -> str:
    '''
    fun(参数 : 类型说明)  -> 返回值类型
    '''
    return 'hello ' + name


# 生成类型别名
Vector = List[float]


def scale(scalar: float, vec: Vector) -> Vector:
    return [num*scalar for num in vec]


# New Type : 辅助函数create不同的类型

NewType("UserId", int)


def get_user_name(user_id: UserId) -> str:
    return "test"


if __name__ == "__main__":

    # quick start
    print(greet_name("wangwei"))

    # 生成类型别名
    print(scale(2.0, [1.0, 2.0, 3.0]))

    # New type
    print(get_user_name(UserId(100))) # 类型检查
    print(get_user_name(10))
