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
		print('funciton: %s  time: %0.3f ret: %s', func.__name__, time2 - time1, ret)
		return ret

	print("33333333333")
	return wrap

@time_decorator
def fun(num):
	print("{}".format(num))
	return num

if __name__ == "__main__":
	fun("34")
