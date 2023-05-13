# -*- coding: utf-8 -*-

import torch
import numpy as np


def test_backward():
    '''
    autograd ：

    torch.autograd.backward(tensors, 
                            grad_tensors = None, 
                            retain_graph= None,
                            create_graph=False) 

    tensors : 求导的张量 
    grad_tensors ： 多个梯度之间，权重的设置
    retain_graph ： 保存计算图
    create_graph ： 创建导数计算图，用于高阶求导

    '''

    # y = （x + w） * (w + 1)  求在 w=1 x=2 处，y对于w的偏导数

    x = torch.tensor([2.], requires_grad=True)
    w = torch.tensor([1.], requires_grad=True)

    print("type x: {}  w: {}".format(x.shape, w.dtype))

    u = torch.add(x, w)
    v = torch.add(w, 1)
    y = torch.mul(u, v)

    # y.backward(retain_graph=True)  #反向传播

    # y.backward()  # 再次执行报错，计算图被释放了，可以设置retain_graph为true 但如果真的再次运行的话，两次运行的偏导值会累加起来

    # print("w对y的偏导数: {}".format(w.grad))  # 2w + x + 1 : 5
    # print("x对y的偏导数: {}".format(x.grad))  # w + 1 : 2


    y0 = torch.mul(u, v)  # y1 = (x + w) * (w + 1)
    y1 = torch.add(u, v)  # y1 = (x + w) + (w + 1)
    
    print("type y0:{}  y1:{} y1 shape:{}".format(y0, y1, y1.shape)) 

    loss = torch.cat([y0, y1], dim = 0 )

    print("loss:{}  loss shape:{}".format(loss, loss.shape))

    grad_tensor = torch.tensor([0.2, 0.8])  # loss包含两种loss分函数，并且这两种函数的对于整体损失的影响，比重还不一样，可以使用grad_tensors计算不同比重情况下损失之和

    loss.backward(gradient=grad_tensor)

    print("w对loss的偏导数: {}".format(w.grad))   # 0.2 * 5 + 0.8 * 2




def test_grad():

    '''
        torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs=None, 
        retain_graph=None, 
        create_graph=False)
    
    功能：求梯度

    outputs: y

    inputs:  x

    返回：inputs 对 outputs的导数， 整体是元组，元素的每个元素是tensor

    '''

    # 求二阶导数
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)

    grad1 = torch.autograd.grad(y, x, create_graph=True)

    print("grad1: {}, grad1[0]: {}".format(grad1, grad1[0]))

    grad2 = torch.autograd.grad(grad1[0], x)   # x对grad1再求一次偏导数,即二阶导数, grad1 必须设置：create_graph

    print("grad2: {}, grad2[0]: {}".format(grad2, grad2[0]))


def lesson01_05():

    # test_backward()

    test_grad()

    return

    

if __name__ == '__main__':

    lesson01_05()
