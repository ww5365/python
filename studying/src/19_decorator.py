import time
''' 
python 闭包机制 使用场景，装饰器

'''

def time_decorator(func):
	print("111111111")
	def wrap(*args):

		print("22222222 {}".format(*args))
		time1 = time.time()
		ret = func(*args)
		time2 = time.time()
		print(f'funciton: {func.__name__}  time: {time2 - time1} ret: {ret}')
		return ret

	print("33333333333")
	return wrap

@time_decorator
def fun(num):
	print("{}".format(num))
	return num

'''
类中@property装饰器作用
 @property 的作用是：把一个方法伪装成“属性”来访问。
这样你可以用 obj.x 的写法触发逻辑，而不是 obj.get_x()。
常见用途：

对外保持“字段式”接口，内部可加计算/校验
只读属性（不提供 setter）
需要时加 setter 做赋值检查
'''

class User:
	def __init__(self, age: int):
		self.age = age  # 会走下面的 setter

	@property
	def age(self) -> int:
		return self._age

	@age.setter
	def age(self, value: int):
		if value < 0:
			raise ValueError("age must be >= 0")
		self._age = value

if __name__ == "__main__":
	fun("34")

	u = User(18)
	u.age = 20  # OK
	# u.age = -1    # ValueError


'''
运行结果：
111111111
33333333333
22222222 34
34
funciton: fun  time: 0.0 ret: 34
'''
