import time
from threading import Thread

''' 
什么是回调？
回调函数是一种编程模式，指将一个函数作为参数传递给另一个函数，并在特定条件满足或事件发生时被调用的函数。简单来说，就是"你完成工作后调用我给你的这个函数"。

为什么要用回调？
异步处理：在非阻塞操作中，当操作完成时通过回调通知
事件驱动：响应特定事件（如点击、数据到达等）
代码解耦：将核心逻辑与具体处理逻辑分离
灵活性：允许动态改变处理行为
可扩展性：易于添加新的处理逻辑

异步回调： 示例
'''

import threading
import time


def async_task(task_name, duration, callback)-> Thread:
    """模拟异步任务"""

    def run_task():
        print(f"开始执行任务: {task_name}")
        time.sleep(duration)
        result = f"{task_name} 完成，耗时 {duration}秒"
        callback(result)

    # 在新线程中执行任务
    thread = threading.Thread(target=run_task)
    thread.start()
    return thread


def task_complete(message):
    """任务完成回调"""
    print(f"回调收到: {message}")


# 启动异步任务
print("主线程继续执行...")
async_task("数据下载", 2, task_complete)
async_task("图片处理", 3, task_complete)
print("所有任务已启动，主线程自由了！")



'''
主线程继续执行...
开始执行任务: 数据下载
开始执行任务: 图片处理所有任务已启动，主线程自由了！

回调收到: 数据下载 完成，耗时 2秒
回调收到: 图片处理 完成，耗时 3秒
'''




