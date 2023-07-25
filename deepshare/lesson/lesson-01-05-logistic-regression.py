# -*- coding: utf-8 -*-

import torch
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(10)

def binary_cross_entropy(y, y_pred):

    t1 = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    
    return t1.mean()

def test_logistic_regression():
    '''
    逻辑回归：
    
    机器学习模型训练的步骤：

    数据

    模型

    损失函数

    优化器

    迭代训练

    '''

    #===========样本数据===================

    sample_nums = 10
    mean_value = 1.7
    bias = 1
    n_data = torch.ones(sample_nums, 2)  # 10 * 2 张量 [10, 2]

    x0 = torch.normal(n_data * mean_value, 1) + bias
    y0 = torch.zeros(sample_nums)  # 类别0的label, 长度为10的向量  [10]

    x1 = torch.normal(-mean_value * n_data, 1) + bias  #类别1， [10, 2]
    y1 = torch.ones(sample_nums)  # 类别1的label, 长度为100的向量  [10]

    train_x = torch.cat([x0, x1], dim = 0)  # 20 * 2
    train_y = torch.cat([y0, y1], dim = 0) # [10] 向量

    # print("n_data tensor: {} trainy: {}".format(train_x, train_y))
    # print("n_data tensor type: {} y0 dtype: {} y0: {}".format(n_data.shape, y0.shape, y0))

    sigmod_fun = nn.Sigmoid()


    #============模型选择========================

    class LR(nn.Module):

        def __init__(self):
            super(LR, self).__init__()   # super() 是调用父类的方法 python2: super(自己类名，self) python3: super()
            self.features = nn.Linear(2, 1)  
            # 输入特征数为2，输出特征数为1  Y = X * W^T +  b
            # 查看模型参数
            # for param in self.features.parameters():
            #   print(param)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):

            x = self.features(x)
            x = self.sigmoid(x)
            return x

    lr_net = LR()    ## 实例化逻辑回归模型

    #==================损失函数=======================

    loss_fn = nn.BCELoss()   # 二分类的交叉熵函数

    #==================优化器========================

    lr = 1e-2   # 学习率
    optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

    #================模型训练=========================

    for epoch in range(2):


        '''
        clone : 深拷贝： 可以返回一个完全相同的tensor,新的tensor开辟新的内存，但是仍然留在计算图中。
                不共享数据内存的同时支持梯度回溯，所以常用在神经网络中某个单元需要重复使用的场景下。
        
        参考：https://zhuanlan.zhihu.com/p/344458484 关键看下clone()的梯度回传

        x = torch.tensor([1.], requires_grad=False)
        x_clone = x.clone()   # 张量x的requires_grad=False, 想通过x_clone进行梯度回溯
        x_clone.requires_grad_() 
        y = x_clone ** 2
        y.backward()
        print(x.grad, x_clone.grad) # 梯度穿向clone的向量x_clone


        inplace原位操作：
        torch.add(x,y, out=y)
        y.add_(x)
        y += x
        y[:] = y + x

        会开辟新内存空间：y最后指向新内存空间
        y = y + x  


        torch.Tensor():  类的构造函数,深拷贝，使用全局默认值构造张量torch.get_default_dtype()
        torch.tensor():  工厂函数, 深拷贝，根据输入推断数据类型dtype

        torch.from_numpy(): 从ndarray中构造张量，浅拷贝，共享内存
        torch.as_tensor():  从python数据结构中构造张量，强拷贝，共享内存
        '''

        ori_w = lr_net.features.weight.clone() 
        ori_b = lr_net.features.bias.clone()

        print("---- epoch: {} train_x: {}  x shape:{} \n ".format(epoch, train_x, train_x.shape))
        print("---- w0: {} w0 shape:{} b0: {} b0 shape:{}".format(lr_net.features.weight, lr_net.features.weight.shape, lr_net.features.bias, lr_net.features.bias.shape))

        for para in lr_net.features.parameters():
            print("---- pargmeter: {}".format(para))

        # 前向传播
        y_pred = lr_net(train_x)   #这里对象是自动调用了forward(self, x) ? 返回：20 * 1 结果

        print("==" * 20)

        #计算loss
        loss = loss_fn(y_pred.squeeze(), train_y)  # 第1个参数：预测值 ()  第2个参数：label
        
        loss1 = binary_cross_entropy(train_y, y_pred.squeeze())  # 自己实现的二分类较差熵损失函数，来验证上面计算的loss

        print("y_pred: {}  y_pred shape: {} loss: {} loss1: {}".format(y_pred, y_pred.shape, loss, loss1))


        # 手动计算了：逻辑回归的反向传播  主要是w b 的梯度  参考：https://blog.csdn.net/chosen1hyj/article/details/93176560
        print("==" * 20)
        t1 = torch.mm(train_x, lr_net.features.weight.t())  # 20 * 1  X*W^T  W : 1 * 2 -> 2 * 1  矩阵乘法
        t2 = torch.add(t1, lr_net.features.bias) #  z = X*W^T + b
        t3 = sigmod_fun(t2) # 20 * 1  a = sigmod(z)
        t4 = t3 - train_y.unsqueeze(dim = 1)   # train_y是20  变成 20 * 1 的相同维度;  a - y 也是： 20 * 1 
        t5 = torch.mul(train_x, t4)  # (20 * 2) * (20 * 1) =>  x * (a - y)  对位乘; dL/dw0 = x0 * (a - y)  dL/dw1 = x1 * (a - y)  dL/db = (a - y)

        print("t1 : {}  t2: {} t3: {} t4:{} t5 : {} t4 mean=db:{} t5 mean=dw: {}".format(t1, t2, t3, t4, t5, t4.mean(), torch.mean(t5, dim = 0))) # 最终的梯度是所有样本的的均值

        optimizer.zero_grad()  #  在反向传播之前，需要清空原来的梯度，不然上个epoch的梯度会累积起来

        # 反向传播
        loss.backward()

        # pytorch平台计算梯度：w,b 对 loss的
        # 模型的反向传播的w和b的梯度，保存在：named_parameters() 返回tuple：参数名  参数的具体值
        for name, params in lr_net.named_parameters():
            print("loss name: {} grad: {}".format(name, params.grad))

        # 更新参数
        optimizer.step()

        # 模型更新完w b 的梯度后 ： w = w - lr * w.grad

        ww = ori_w - (lr * lr_net.features.weight.grad)   # 手动计算
        bb = ori_b - (lr * lr_net.features.bias.grad)
        print("manual w : {}".format(ww))
        print("manual b : {}".format(bb))
        print("backward grad: ori_w = {}, w= {}  ori_b = {}, b= {}".format(ori_w, lr_net.features.weight, ori_b, lr_net.features.bias))

        # 绘图看效果
        if  epoch % 10 == 0:
            mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类, 大于0.5的变成1.(浮点数),同时降维 20 * 1 -> 20

            correct = (mask == train_y).sum()  # 张量中，对位元素相同的个数统计;torch.int64 的标量;预测为1/0且实际为1/0的个数,也就是预测正确的个数

            acc = correct.item()/train_y.size(0) # 计算准确率   
            # tensor.item() 返回一个数，仅适用张量只有1个元素的情况
            # Tensor.size(dim=None) → torch.Size or int  指明dim，返回特定维度值； 否则，返回的tuple

            plt.scatter(x0.data.numpy()[:, 0],x0.data.numpy()[:, 1], c = 'r', label = "class 0") # 两个维度特征(x1, x2),标注到xy坐标系上
            plt.scatter(x1.data.numpy()[:, 0],x1.data.numpy()[:, 1], c = 'b', label = "class 1")

            # print("==" * 40)

            # print("self.features: {} type:{}".format(lr_net.features.weight, lr_net.features.weight.shape))  # weight: 1 * 2
            # print("self.features: {} type:{}".format(lr_net.features.weight[0], lr_net.features.weight[0].shape))
            # print("self.features: {} type:{} bias:{}".format(lr_net.features.bias, lr_net.features.bias.shape, lr_net.features.bias[0]))
            
            print("==" * 40)


            w0, w1 = lr_net.features.weight[0]

            w0, w1 = float(w0.item()), float(w1.item())

            plot_b = float(lr_net.features.bias[0].item())

            plot_x = np.arange(-6, 6, 0.1)  # range 和 arange 区别： 都不包含end，arrange步长支持小数

            plot_y = (-w0 * plot_x - plot_b) / w1  ## 这条线，画出来？ 是什么原理？ w0*x0 + w1*x1 + b = 0

            # 画线参考：https://blog.csdn.net/weixin_43772533/article/details/100974008


            plt.xlim(-5, 7)
            plt.ylim(-7, 7)
            plt.plot(plot_x, plot_y)

            plt.text(-5, 5, 'loss: %.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})

            plt.title("epoch: {}\n w0:{:.2f}, w1:{:.2f}, b:{:.2f}, acc:{:.2f}".format(epoch, w0, w1, plot_b, acc))

            plt.legend()
            plt.show()
            # plt.pause(0.5)

            if acc > 0.99:
                break
            

def lesson01_05():

    test_logistic_regression()

    return


if __name__ == '__main__':

    lesson01_05()
    
    y = torch.tensor(
    [[0.3043],
        [0.2677],
        [0.1732],
        [0.2540]]
    )

    print("y shape:{} type: {}".format(y.shape, y.dtype))

    print("y shape:{} ".format(y.ge(0.2).float().squeeze()))

    x1 = torch.tensor([1.2, 3.2, 1.0])
    x2 = torch.tensor([1.3, 3.2, 1.0])

    sum = (x1==x2).sum()   

    print("x1 == x2: {}  sum: {} type:{}".format(x1.size(), (x1==x2).sum(), sum.dtype))
