#-*- encoding:utf8 -*-

import numpy as np

if __name__ == '__main__':
    L = [1,2,3,4,5,6]
    print ("python list:", L, type(L))
    vec = np.array(L)
    print("vector:", vec, type(vec), vec.shape)
    
    b = np.array([[1,2,3],[5,6,7],[7,8,9],[2,3,4]])
    print("matrix:", b, b.shape,type(b))

    ## reshape后的总个数 = 之前的总个数，也就是：row * clo = row2 * clo2 
    c = b.reshape((6, -1))
    print("c reshape: ", c)
    b[0][0] = 13

    print("c = ", c, c.dtype)

    d = np.array([[1,2,3],[4,5,6]], dtype = np.float)
    ##复数
    f = np.array([[1,2,3],[4,5,6]], dtype = np.complex)
    print("d = ", d)
    print("f = ", f)
    ##改变元素数据类型
    ff = f.astype(np.int)
    print("ff = ", ff)


    



