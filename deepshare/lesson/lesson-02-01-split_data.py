# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import shutil  # shutil模块是对os模块的补充，主要针对文件的拷贝、删除、移动、压缩和解压操作
import random  # 使用random.shuffle(x)


BASE_DIR = os.path.dirname(__file__)  # 当前文件的根目录
data_dir = os.path.join(BASE_DIR, "..", "data", "RMB_data")  # .\deepshare\lesson\..\data\RMB_data
split_dir = os.path.join(BASE_DIR, "..", "data", "rmb_split")  # .\deepshare\lesson\..\data\rmb_split

'''
数据集进行划分：
train : ./data/rmb_split/1/*.jpg
valid
test
'''

def mkdir(dir_name) -> None:
    if not os.path.exists(dir_name):
        # os.mkdir(dir_name)  # 创建单层目录
        os.makedirs(dir_name) # 创建多层目录

def split_dataset() -> None:

    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")
    
    if not os.path.exists(data_dir):
        raise Exception("{} dir not exists, please download RMB_data.rar put {}".format(data_dir, os.path.dirname(data_dir)))

    train_per = 0.8
    test_per = 0.1
    valid_per = 0.1
    
    for root, dirs, files in os.walk(data_dir):
        # print("root: {} dirs: {} files: {}".format(root, dirs, files))  # root: ..\data\RMB_data  dirs: 1 100 files:是文件夹1 10下面的所有图片
        # 只针对目录下的子目录进行遍历： 即：目录1 100
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))   #返回这个目录下面所有文件名
            # print("sub dir files: {}".format(imgs))
            imgs = list(filter(lambda x : x.endswith('.jpg'), imgs)) 
             # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换
            random.shuffle(imgs)  # 文件列表做shuffle
            img_count = len(imgs)
            print("sub dir : {}, imgs counts: {}".format(os.path.join(root,sub_dir), img_count))
            train_cnt = int(img_count * train_per)
            valid_cnt = int(img_count * valid_per)
            for i in range(img_count):
                if i < train_cnt:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < train_cnt + valid_cnt:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                mkdir(out_dir)
                src_file = os.path.join(root, sub_dir, imgs[i])
                target_file = os.path.join(out_dir, imgs[i])
                shutil.copy(src_file, target_file)  # shutil  文件和文件集合的高阶操作,特别是提供了一些支持文件拷贝和删除的函数
            
            print("sub_dir: {}, train: {} valid:{} test:{}".format(sub_dir, train_cnt, valid_cnt, (img_count - train_cnt - valid_cnt)))
    return

def lesson02_01() -> None:

    split_dataset()  # 将数据划分为训练集，测试集，验证集
    return


if __name__ == '__main__':

    # lesson02_01()
    print("BASE DIR: {}".format(BASE_DIR))


