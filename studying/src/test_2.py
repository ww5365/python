import sys
import os
import collections
import math
import numpy as np
from multiprocessing import Array

import torch 


if __name__ == '__main__':
    
    # torch.log默认的底数是? e
    
    print("tensor log: ", torch.log(torch.tensor([4.])))
    

    # 前4或8位是否在dict的key中
    
    t1 = torch.randint(1, 10, size=(10, ))
    
    print("[low, high) 均匀分布: {}".format(t1))
    
    print("[low, high) 均匀分布: {}".format(torch.randint_like(t1, 3)))
    
    
    print("random 排列: {}".format(torch.randperm(10)))
    
    print("bernuli 分布: {}".format(torch.bernoulli(torch.ones((3,3)))))
    
    t3 = torch.ones(2, 3)
    
    print("t3: {}".format(t3))
    
    t4 = torch.stack([t3,t3,t3], dim=2)
    
    print("t4: {}".format(t4))
    
    
    t5 = torch.tensor([1., 2.])
    
    print("t5.shape: {}".format(t5.shape))
    
    t6 = torch.chunk(torch.ones((2,9)), chunks=4, dim=0)
    
    for idx, t in enumerate(t6):
        print("t6 idx: {}  t: {}  shape: {}".format(idx, t, t.shape))
    