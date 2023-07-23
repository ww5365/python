# -*- coding: utf-8 -*-

# 可参考：https://gitee.com/zhaohui24/pytorch-framework

import torch
import numpy as np
import os

import matplotlib.pyplot as plt

torch.manual_seed(10)  # 设置了随机数种子，下次再次运行本py文件，torch.rand生成的数是一样的;这样多次运行可以复现

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def linear_regression():

    # 创建训练数据
    x = torch.rand((20,1)) * 10   # shape= (20,1) 张量 [0, 10) 均匀分布
    y = 2 * x + (torch.randn(20,1) + 5)  # y  = 2 * x + b

    print("x: {} xsize: {}".format(x, x.shape))
    print("y: {} ysize: {}".format(y, y.shape))

    # 学习线性模型

    # 定义学习参数
    w = torch.randn((1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    print("w: {} w.shape: {}".format(w, w.shape))
    print("b: {} b.shape: {}".format(b, b.shape))
    
    lr = 1e-2

    for epoch in range(20):

        # print("w.grade {}".format(w.grad))
        if w.grad is not None:
            # print("before w grad: {}".format(w.grad))
            w.grad.zero_()  # 把上轮的梯度清空
            # print("w.grade {}".format(w.grad))

        if b.grad is not None:
            # print("before b grad: {}".format(b.grad))
            b.grad.zero_()

        # 前向
        # y_pred2 = w * x + b
        # print("y_pred2: {} y_pred2.size: {}".format(y_pred2, y_pred2.shape))
        wx = torch.mul(w, x)
        y_pred = torch.add(wx, b)  # 和y_pred2结果一致
        # print("y_pred: {} y_pred.size: {}".format(y_pred, y_pred.shape))

        # 损失
        loss = (0.5 * (y_pred - y) ** 2).mean()

        # 后向
        loss.backward()

        # 参数更新，梯度下降

        # print("w grad: {}".format(w.grad))
        # print("b grad: {}".format(b.grad))
    

        
        w.data.sub_(w.grad * lr)   # 可以
        b.data.sub_(b.grad * lr)
        
        # other = w.data   # 创建了一个新的tesor other；有两个特点：1. 和w是共享内存的  2.other是不可求导的 requires_grad=False        
        # print("other value: {}".format(other))
        
        # w.data -= lr * w.grad  # 可以   w.data 不会影响w的叶子节点属性，同时能修改w的值
        # b.data -= lr * b.grad

        #with torch.no_grad():
        # w = w - w.grad * lr  # 不可以
        # b = b - b.grad * lr
        #     # 计算完了，w b 梯度就没有， 很奇怪？ 可以这样更新梯度吗？
        # 叶子节点，在创建的时候requires_grad = True 不支持in place操作
        # 同时，上面公式会破坏计算图，使得w，b变成中间节点，w,b的存储地址会较创建它们时候的地址发生改变；中间节点不会保存其梯度，所以无法进行更新;
        # 所以不能使用w = w - lr * w.grad 
        # 可以使用w.data = w.data – w.grad*lr / w.data.sub_(lr * w.grad) 原因如下：
        # out = x .data 返回和 x 的相同数据 tensor,而且这个新的tensor out和原来的tensor x是共用数据的，一者改变，另一者也会跟着改变，而且新分离得到的tensor的require s_grad = False, 即不可求导的
        # 相当于detach操作了

        # 参考：https://zhuanlan.zhihu.com/p/557069097

        # print("w grad: {}".format(w.grad))
        # print("b grad: {}".format(b.grad))

        # print("w : {}".format(w))
        # print("b : {}".format(b))

        # print("epoch: {} loss : {}".format(epoch, loss))

 
        

        # if epoch % 20 == 0:

        #     plt.scatter(x.data.numpy(), y.data.numpy())
        #     plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        #     plt.text(2, 20, 'loss=%.4f'%loss.data.numpy())
        #     plt.title("epoch: {}\n w:{} b:{}".format(epoch, w.data.numpy(), b.data.numpy()))
        #     plt.pause(1)

        
        print("loss data: {} type: {}  {}".format(loss.data.numpy(), type(loss.data), type(loss.data.numpy())))
        print("address: {}  {}".format(id(loss), id(loss.data)))
        if loss.data.numpy() < 0.5:
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), y_pred.data.numpy(), 'b-', lw=3)
            plt.text(2, 20, 'loss=%.4f'%loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.title("epoch: {}\n w:{} b:{}".format(epoch, w.data.numpy(), b.data.numpy()))
            plt.show()
            break



if __name__ == '__main__':


    linear_regression()
