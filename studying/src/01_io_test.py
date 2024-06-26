# -*- coding: utf8 -*-

import os
import sys
import random
import pathlib
import time
from datetime import datetime
import types
import codecs
import tarfile
import chardet
import asyncio


'''
dir():  dir([object])  --object: 对象、变量、类型  返回：属性列表list


dir() 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；
      带参数时，返回参数的属性、方法列表。
      如果参数包含方法__dir__()，该方法将被调用。
      如果参数不包含__dir__()，该方法将最大限度地收集参数信息。

'''


class Foo(object):
    def __init__(self):
        __num = ""

    def public(self):
        print("public funciton")

    def __private(self):
        print("private function")


def get_file_path():
    '''
      os.getcwd(): 
      os.path.dirname: 返回当前文件所在路径
      os.path.abspath: 返回当前路径的绝对路径
      os.path.split: 按照"\"来切割最后一层目录和之前的路径,tuple：比如： /tmp/test/test1  ->  (/tmp/test, test1)
      os.path.join: 将两个目录字符串拼接起来，用/进行连接
    '''

    # 获取当前运行进程的工作目录
    print("os.getcwd(): ", os.getcwd())
    print("abspath:", os.path.abspath(os.getcwd()))

    # 当前文件的路径： d:/workspace/python/studying/src linux风格
    print("dirname:", os.path.dirname(__file__))
    # 绝对路径：  d:\workspace\python\studying\src windows路径风格
    cur_path = os.path.abspath(os.path.dirname(__file__))
    print("cur_path:", cur_path)
    print("split path:", os.path.split(cur_path))

    root_path = os.path.split(cur_path)[0]
    what_path = os.path.split(root_path)[0]  # 获取文件目录，上两层目录路径
    print("root_path:", root_path, what_path)

    work_path = os.path.join(root_path, 'Assignment3-1')
    print("work_path: ", work_path)

    # 使用pathlib直接获取文件的root路径,并加入到系统path中
    print("pathlib path: ", pathlib.Path(__file__))  # window路径风格：
    root_path = pathlib.Path(
        __file__).parent.parent.absolute()  # 父父节点的绝对路径，感觉更好用
    print("pathlib parent: ", root_path)
    sys.path.append(sys.path.append(root_path))  # 加入到系统搜索路径
    print("pathlib and  syspath:", root_path, sys.path)

    '''
     目标：遍历某个目录下所有文件和子目录?

        os.walk : 通过在目录树中游走输出在目录中的文件名，向上或者向下;
        os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
        输入和输出说明：

        top -- 是你所要遍历的目录的地址

        返回的是一个三元组(root,dirs,files)。
            root 所指的是当前正在遍历的这个文件夹的本身的地址
            dirs 是一个 list ，内容是该文件夹中所有的子目录的名字(不包括子子目录)
            files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。如果 topdown 参数为 True，walk会先遍历top文件夹，与top 文件夹中每一个子目录。
        onerror -- 可选，需要一个 callable 对象，当 walk 需要异常时，会调用。
        followlinks -- 可选，如果为 True，则会遍历目录下的快捷方式(linux 下是软连接 symbolic link )实际所指的目录(默认关闭)，如果为 False，则优先遍历 top 的子目录。
    '''

    '''
        root_path: d:\workspace\python\studying
        总共循环几次：
        第一轮： data src test.txt
        第二轮： 子目录data下面的文件
        第三轮：子目录data/sample下面的文件和目录
        第三轮：子目录src下面的文件和目录
    '''

    level = 0
    for root, dirs, files in os.walk(root_path, topdown=True):
        print("loop dir level:", level, root)
        level += 1
        for name in dirs:
            print("current dir subdirs: ", os.path.join(root, name))

        for name in files:
            print("current dir files: ", os.path.join(root, name))


def get_file_name():
    '''
    file = ".\dir\test.txt"
    os.path.split : 返回(.\dir, test.txt)
    os.path.splitext : 返回(.\dir\test, txt)

    '''
    CUR_PATH = os.path.abspath(os.path.dirname(__file__))

    # 获取文件的名称，去掉后缀
    (path_filename, ext) = os.path.splitext(__file__)
    (path, filename) = os.path.split(path_filename)

    new_file_name = CUR_PATH + '\\' + filename + "_with_label" + ext

    print("filename: ", filename, path, path_filename)
    print("new filename: ", new_file_name)


def read_file():
    '''
    1、几种不同的文件读取的方式
    '''
    # 1 正常思路: 无异常处理
    root_path = pathlib.Path(__file__).parent
    file_path = os.path.join(root_path, 'test.txt')
    print("file_path:", file_path)
    f = open(file_path, mode='r+', encoding="utf-8")
    lines = f.readlines()  # 全部读入，放入list 包含换行符
    print("file content:", lines)

    f.seek(0)  # 返回文件头
    line = f.readline()  # 读入文件的1行
    line_no = 0
    while line:
        print("line:", line)
        line = f.readline()

    f.close()  # 出异常，不会释放fp

    # 2 比较好的的思路： 有异常处理
    f2 = open(file_path)
    try:
        for line in f2.readlines():
            print("line:", line)
    except:
        print("open error")  # IOError
    finally:
        f2.close()  # 保证file会释放

    # 3 最优方案： 自带异常处理，不用close；相当于方案2，但更加精炼
    with open(file_path, 'r', encoding='utf-8') as f3:

        # readlines
        for line in f3.readlines():
            print("line3:", line)

        # readline
        f3.seek(0)
        line = f3.readline()
        while line:
            line_no = line_no + 1
            print("line33:%d -> %s" % (line_no, line))
            line = f3.readline()

        # 直接读取
        f3.seek(0)
        line_no = 0
        for line in f3:
            line = line.split()
            print("line333: %d -> %s" % (line_no, line))


def save_file():
    '''
    os.path.exists: 目录是否存在
    os.makedirs: 创建目录
    file.write: 写入文件
    '''

    # 获取当前文件目录
    cur_dir = os.path.dirname(__file__)
    print(cur_dir)

    # 创建tmp目录
    full_dir = os.path.join(cur_dir, "tmp/")
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
        print("full path: ", full_dir)

    file_path = full_dir + "random.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(10):
            f.write("\t".join([str(i), random.choice(['a', 'b', 'c'])]))
            f.write('\n')


def codecs_use():

    # python3 字符串存储，编码相关
    str1 = '王伟'   # type: str
    u_str1 = str1.encode('gb18030')  # 以gb18030编码对str进行编码，获得bytes类型对象
    print("encode typpe: ", u_str1, type(u_str1))
    print(u_str1.decode('gb18030'))
    # print(str1.decode('utf-8'))

    print("--" * 30)
    cur_path = os.path.abspath(os.path.dirname(__file__))
    work_path = pathlib.Path(cur_path).absolute().parent
    print("workp path: ", work_path)

    '''
    os.path.join()函数：连接两个或更多的路径名组件
        1.如果各组件名首字母不包含’/’，则函数会自动加上
        2.如果有一个组件是一个绝对路径，之前的所有组件均会被舍，
            比如：os.path.join(d:/test1/test2, /test3/test4) -> d:/test3/test4 第二个参数是绝对路径，将test1 ，test2舍弃
        3.如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾
    '''
    file_path = os.path.join(work_path,
                             r'data\sample\random.txt')  # 得到错误路径：d:\tmp\random.txt
    print("file path: ", file_path)

    # codecs 库主要可以读入文件的编码格式
    f = codecs.open(file_path, 'wb+')  # 以二进制的方式读写

    # 写入types类型的数据
    f.write(u_str1)

    # 以bytes方式读取文件数据
    f.seek(0)
    content = f.read()  # 取文件内容时候，会自动转换为内部的unicode, types类型
    print("content: ", content, type(content), content.decode("gb18030"))

    encode_info = chardet.detect(content)  # 只能检测bytes类型的编码格式
    print(encode_info)
    print("content2:", content.decode(encode_info["encoding"]))

    f.close()

def unzip_file():
    '''
    使用tarfile读取压缩文件中的多个文本文件
    比如：
    test.tar.gz
        file1
        file2
        file3
    '''
    data_path = "c:\dir\test"
    vip_siteid_set = {}
    siteid_set = set()
    siteid2id_dict = dict()
    for root, dirs, files in os.walk(data_path, topdown=False):
        for tar_file in files:
            if tar_file.endswith("data2.tar.gz"):
                tar_data_path = os.path.join(data_path, tar_file)
                # print(data_path)
                tar = tarfile.open(tar_data_path, "r:gz")  # 打开压缩包: tar.gz
                for member in tar.getmembers():  # 逐个解压压缩包中的各个文件; 返回TarInfo对象，里面有name属性
                    # print(member)
                    f = tar.extractfile(member)  # 提取压缩包中的文件;异常文件，可能返回None
                    if f is not None:
                        for binary in f.readlines():  # 逐行读取文件, 返回的是bytes
                            # print(binary)
                            content = binary.decode(
                                "utf-8").strip()  # 将bytes数据转成str
                            line = content.split('\x01')
                            # print("line: ", line)
                            if len(line) < 3:
                                continue
                            site_id = line[0]
                            id = line[1]
                            if site_id in vip_siteid_set:
                                siteid_set.add(site_id)
                                if site_id not in siteid2id_dict:
                                    siteid2id_dict[site_id] = id
                tar.close()  # 处理完一个压缩
    return

@asyncio.coroutine
def countDown(name,num):
    while num>0:
        print(f'countdown[{name}]:{num}')  # {name} 被替换为相应变量的值;去掉f，不会被替换
        yield from asyncio.sleep(1)
        num -= 1

def basic_input():
    # 接受键盘输入
    str_info = input("input your tips info: ")
    print(str_info)

    # print(f"{var}") : formatted string literal 以f开头，包含的{}表达式在程序运行时会被表达式的值代替。
    loop = asyncio.get_event_loop()
    tasks =[countDown("A",10),countDown("B",5),]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close



if __name__ == '__main__':

    # 获取键盘上用户输入：input
    basic_input()

   # # dir function use case
   # print("\n".join(dir(Foo)))  # 4个成员函数

    # get_file_path()
   # read_file()
   # save_file()
    # codecs_use()

   # # get filename
   # get_file_name()

   # # 格式化打印
   # str1 = "wang"
   # print("format print: %s ... %s" % (str1, str1), end='*')

   # print("\n")
   # print("escape charater: \\")

   # # time 使用
   # start = datetime.now()
   # time.sleep(1)
   # print((datetime.now() - start).seconds)

   # # 格式输出，还能拼接
   # format_str = "test format %d : %s"
   # num = 10
   # str1 = "format str"
   # format_str = format_str % (num, str1)
   # print(format_str)
   # print("test format2: %d : %s" % (num, str1))

   # # str.format  一种格式化字符串的函数,它通过 {} 和 : 来代替以前的 %
   # wiki_url = "http://www.baidu.com/api/{}/query={}"
   # url = wiki_url.format("v1", "kaocheng")  # 直接填{}中内容
   # print(url)

   # wiki_url2 = "http://www.baidu.com/api/{version}/query={key}"
   # url2 = wiki_url2.format(version="v2", key="kaocheng")  # 使用{参数}
   # print(url2)

   # # 保留小数点后两位
   # print("{:.2f}".format(3.1415926))

   # # 四舍五入，保留小数后n位 round(x, n)
   # x = 1.545
   # print(round(x, 2))
   # print(round(x))  # 默认保留整数部分

   # # 数据类型

   # li2 = [1, 2, 3]

   # print("list type: ", type(li2), type(li2) == list)
