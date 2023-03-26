# -*- coding:utf-8 -*-

import torch 
import numpy as np

torch.set_printoptions(edgeitems=3) # 设置打印时，隐藏的情况；显示上下三行


'''
X = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                    33.9, 21.8, 48.4, 60.4, 68.4]
Y = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                    3.0, -4.0, 6.0, 13.0, 21.0]

用线性回归模型来拟合上面的数据实例：
y = w*x + b

损失函数使用：MSE

'''

# 定义模型
def model(X, w, b):
    return w*X + b

# 定义损失函数:均方误差
def mse_loss(Y, y_pred):
    square_loss = (Y - y_pred) ** 2
    return square_loss.mean()

# 定义loss function对y_pred的导数
def d_loss_function(Y, y_pred):
    N = y_pred.size(0)
    result = 2 * (y_pred-Y) * 1/N
    return result


# y_pred函数对各个参数变量的梯度，即导数
def d_y_pred_w(X, w, b):
    return X

def d_y_pred_b(X, w, b):
    return 1.0

# 综合计算损失函数对各参数的梯度
def grad_fun(X, Y, y_pred, w, b):
    
    d_loss_pred = d_loss_function(Y, y_pred)
    d_pred_w = d_y_pred_w(X, w, b) * d_loss_pred
    d_pred_b = d_y_pred_b(X, w, b) * d_loss_pred

    return torch.stack([d_pred_w.sum(), d_pred_b.sum()])   ## 所有样本梯度的和？


# 训练过程

def training(epochs, lr, params, X, Y):

    for epoch in range(1, epochs + 1):
        w, b = params

        y_pred = model(X, w, b)

        loss = mse_loss(Y, y_pred)

        grad = grad_fun(X, Y, y_pred, w, b)
        
        print("before Params: {}".format(params))
        
        print("grad: {}".format(grad))

        params = params - grad * lr

        print("Params: {}".format(params))
        print("Epoch %d, Loss %f" %(epoch, loss))

    print ("final params: {}".format(params))
    return params

def linear_regression_demo1(X, Y):
    
    training(epochs = 20000, lr = 1e-4, params = torch.tensor([1.0, 0.0]), X = X, Y = Y)
    return



def main():
    X = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    Y = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])

    torch.Tensor([1, 2])

    print(X.size())
    print(X.size(0))
    
    # X = 0.1 * X   ## 这个也很关键啊，不规范化，训练不出结果来,loss 越来越大？ 原因是由于grad值太大，导致learn_rate 不足够小了，每个学习步长太大了，不收敛了 

    linear_regression_demo1(X, Y)



if __name__ == "__main__":
    main()
