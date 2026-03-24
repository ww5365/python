

'''

@dataclass 的作用：自动帮你生成“数据类”常见样板代码，让“只存数据”的类更简洁。

最常用的自动生成内容：

__init__：按字段自动生成构造函数
__repr__：打印对象更友好
__eq__：按字段比较对象是否相等
可选 order=True：支持大小比较
可选 frozen=True：对象不可变（类似只读）


@dataclass 本质是一个 **运行时装饰器函数** ，在类定义完成后“改造”这个类。

实现思路可以概括成 4 步：

读取类上的 __annotations__，拿到字段定义和顺序
解析每个字段（默认值、field(...)、init/repr/compare 等）
动态生成方法（__init__ / __repr__ / __eq__ / 可选 __hash__ / 排序方法）
把这些方法挂回类对象并返回

'''


def mini_dataclass(cls):
    '''
    这个演示了最小的运行时装饰器的实现：dataclass
    '''

    test1 = getattr(cls, "__annotations__", {})
    print(f"test1: {test1}")

    fields = list(getattr(cls, "__annotations__", {}).keys())

    def __init__(self, *args, **kwargs):
        for i, name in enumerate(fields):
            if i < len(args):
                value = args[i]
            else:
                value = kwargs.get(name, getattr(cls, name, None))
            setattr(self, name, value)

    def __repr__(self):
        body = ", ".join(f"{n}={getattr(self, n)!r}" for n in fields)
        return f"{cls.__name__}({body})"

    def __eq__(self, other):
        if type(other) is not cls:
            return NotImplemented
        return all(getattr(self, n) == getattr(other, n) for n in fields)

    cls.__init__ = __init__
    cls.__repr__ = __repr__
    cls.__eq__ = __eq__
    return cls


@mini_dataclass
class User:
    name: str
    age: int
    city: str = "Shenzhen"  # 类属性默认值

'''
cls是类本身，上面的修饰，等价的是：User = mini_dataclass(User)
'''

u1 = User("Alice", 18)
u2 = User(name="Alice", age=18)
u3 = User("Bob", 20, "Beijing")


print("======================")

print(u1)          # User(name='Alice', age=18, city='Shenzhen')
print(u3)          # User(name='Bob', age=20, city='Beijing')
print(u1 == u2)    # True
print(u1 == u3)    # False

print("======================")