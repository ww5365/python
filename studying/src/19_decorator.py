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

if __name__ == "__main__":
	fun("34")


'''
运行结果：
111111111
33333333333
22222222 34
34
funciton: fun  time: 0.0 ret: 34
'''
