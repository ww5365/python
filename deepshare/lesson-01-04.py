# -*- coding: utf-8 -*-

import torch
import numpy as np


def lesson01_04() -> None:

    '''
    torch.tensor 的属性:

    is_leaf: 是否为叶子节点，计算图中，叶子节点很关键

    grad:  记录了张量的梯度

    grad_fn: 记录了张量被创建时，所用到的加减乘除法


    静态图和动态图


    '''

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    y1 = y.tanh



    a.retain_grad()
    y.backward()

    # 查看叶子节点
    print("is leaf: \n", w.is_leaf, w.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)  # True True False False False

    #  反向传播后, 张量是否还保留梯度? 非叶子节点没有了; 
    #  想保留非叶子节点的梯度，怎么办？  a.retain_grad()
    print("grad: \n", w.grad , x.grad, a.grad, b.grad, y.grad)  
    # y = (w + x) * (w + 1)  w对y的梯度 x对y的梯度  tensor([5.]) tensor([2.]) tensor([2.]) None None

    # 查看 grad_fn
    print("grad: \n", w.grad_fn , x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)  # None None <AddBackward0 object at 0x000001CD5649FF40> <AddBackward0 object at 0x000001CD5649FF70> <MulBackward0 object at 0x000001CD5649FEB0>

    # 叶子节点：没有grad_fn   非叶子节点：a = w + x  是使用了加法生成的，所以张量a此时记录的grad_fn: addBackward  y是：multibackward


    return

    

if __name__ == '__main__':

    lesson01_04()
