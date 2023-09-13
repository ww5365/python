# -*- encoding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


def torch_use():
    
    
    
    '''
    
    convert_tokens_to_ids(seq): 将 token 映射为 id
    tokenizer.convert_tokens_to_ids(['[CLS]', 'Hello', 'word', '!', '[SEP]'])
    [101, 8667, 1937, 106, 102]
   
    convert_ids_to_tokens(ids, skip_special_tokens)：将 id 映射为 token
    tokenizer.convert_ids_to_tokens(tokens)
    ['[CLS]', 'Hello', 'word', '!', '[SEP]'] 
    
    
    利用Tokenizer对多句进行分词和（标准化）编码
    encode_plus返回所有编码信息
    In:sen_code = tokenizer.encode_plus("i like you", "but not him")
    Out : 
        {'input_ids': [101, 1045, 2066, 2017, 102, 2021, 2025, 2032, 102],
        'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1],
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}   
    
    '''
    
    
    
    
    '''
    
    input_ids 就是一连串 token 在字典中的对应id。形状为 (batch_size, sequence_length)。
    Bert 的输入需要用 [CLS] 和 [SEP] 进行标记，开头用 [CLS]，句子结尾用 [SEP]，各类bert模型对应的输入格式如下所示：
    bert:       [CLS] + tokens + [SEP] + padding
    roberta:    [CLS] + prefix_space + tokens + [SEP] + padding
    distilbert: [CLS] + tokens + [SEP] + padding
    xlm:        [CLS] + tokens + [SEP] + padding
    xlnet:      padding + tokens + [SEP] + [CLS]
    
    token_type_ids 可选。就是 token 对应的句子id，值为0或1（0表示对应的token属于第一句，1表示属于第二句）。
    形状为(batch_size, sequence_length)。如为None则BertModel会默认全为0（即a句）。
   
    tokens：[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    token_type_ids：0   0  0    0    0     0       0   0   1  1  1  1   1   1
    tokens：[CLS] the dog is hairy . [SEP]
    token_type_ids：0   0   0   0  0     0   0
    
    '''
    
    text = "CLS"


def test_nn_embedding():

    corpus = [[1,2,3,4],[2,3,4,0],[5,6,7,7]]  # 模拟['i like china to', 'how you are' , 'i like ha ha']
    emb_input = torch.LongTensor(corpus)
    print(emb_input)

    embeding = torch.nn.Embedding(10, 2) # 有10个单词的词表，每个单词用4维向量表示

    print("nn.embeding weight: {}".format(embeding.weight)) # 初始是randn的
    
    x = embeding(emb_input) # [3,4] -> [3, 4, 2] 通过input索引位置，查找embedding词表

    print("x: {} x shape: {}".format(x, x.shape))

    emb_input2 = torch.transpose(emb_input, dim0= 0 , dim1=1)
    print("id1: {} id2: {}".format(id(emb_input), id(emb_input2)))
    x2 = embeding(emb_input2)
    print("x2: {} x2 shape: {}".format(x2, x2.shape)) # [4, 3, 2]


def test_F_funtion():

    ## F.linear 使用 xA^T
    corpus = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                [[13,14,15,16],[17,18,19,20],[21,22,23,24]]]) #2*3*4
    weight = torch.tensor([[1,2],[3,4],[5,6],[7,8]])  # 4*2
    print("shape: {}  {}".format(corpus.shape, weight.shape))
    result = F.linear(corpus, weight.T)  # 2*3*2
    print("result: {} shape:{} size:{}".format(result, result.shape, result.size()))

    # F.linear  
    test1 = torch.tensor([[[0.5, 0.5],[0.5, 0.6]],[[0.5, 0.5], [0.6, 0.5]]]) # 2*2*2
    w = torch.tensor([[1,1,1,1],[2,2,2,2]]) 
    w = w.to(torch.float) # 2 * 4
    bias = torch.tensor([0.1])
    res3 = F.linear(test1, weight=w.T, bias= bias) # 2 * 2 * 2 @ 2 * 4 => 2 * 2 * 4
    print("res3: {} shape:{}".format(res3, res3.size()))

    # softmax 三维，针对某1维度做softmax
    print("test1 shape: {}".format(test1.size()))
    res1 = F.softmax(test1, dim = 0)  # 第0维度，全部8个值进行softmax计算
    res1 = F.softmax(test1, dim = 1)  # 第1维度，有2个二维矩阵，每个矩阵按照列进行计算，2个元素
    res1 = F.softmax(test1, dim = 2)  # 第2维度，有2个二维矩阵，每个矩阵按照行进行计算，2个元素
    print("res1: {}   shape: {}".format(res1, res1.size()))

    # dropout 三个维度做dropout，是什么情况？
    '''
    1、dropout由于会改变输入数据的均值，所以需要对权重进行改变;会将没有屏蔽的数值进行调整(缩放)乘以1/(1−p)
    测试时：可以不用resacle输入了
    2. 3维度张量的情况下，是对全部的元素，以p的概率，伯努利，进行失活
    参考：https://www.cnblogs.com/CircleWang/p/16025723.html
    '''
    res2  = F.dropout(test1, p=0.5, training=True)
    print("res2: {} shape:{}".format(res2, res2.size()))

    # 2 * 2 * 2 按照矩阵维度,dim=0有2个，计算对应位置上的元素的均值：4个位置，每个位置上2个值平均
    res4 = test1.sum(dim=0)/2
    print("res4: {} shape:{}".format(res4, res4.size()))

    # F.relu
    test3 = torch.tensor([[0.5, 1.3],[-2, 0.4]])
    res5 = F.relu(test3)
    print("res5: {} shape:{}".format(res5, res5.size())) # 2*2 [[0.5, 1.3],[0, 0.4]]
    

    # test1 = torch.randn([3,2,3])


def test_contigous():
    '''
    tensor变量调用contiguous()函数会使tensor变量在内存中的存储变得连续。
    contiguous()：view只能用在contiguous的variable上, 一般view之前需先contiguous操作
    '''
    x = torch.ones(3, 2)
    print(x.is_contiguous()) # true
    xx = torch.transpose(x, 0, 1)
    print(xx.is_contiguous()) # false
    xxx = xx.contiguous()
    print(xxx.is_contiguous()) #True

    print(xxx.view(6, 1)) # view 之前必須是contiguous的张量

    # PyTorch在0.4版本以后提供了reshape方法，实现了类似于 tensor.contigous().view(*args)

    print(x.reshape(6,1))


def test_masked_filled():

    '''
    masked_fill方法有两个参数，masked和value
    masked是Tensor，元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置

    t.masked_fill(m, -1e9)
    参数m必须与t的size相同或者两者是可广播(broadcasting-semantics)
    '''

    t = torch.randint(0,2, (3,2)).to(torch.float)  # 3 * 2
    m = torch.tensor([[True],[False],[False]]).to(torch.float)

    print("m: {}  t:{}".format(m==0, t))
    res = t.masked_fill(m==0, float('-inf'))

    print("masked_fill: {}".format(res))

    ## masked attetion中对角掩码矩阵生成

    '''
    torch.triu(input, diagonal=0, out=None) → Tensor
    返回矩阵上三角部分，其余部分定义为0。
    '''
    sz = 4
    print(torch.triu(torch.ones(sz, sz)))
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # 转换成下三角矩阵
    print("mask: {}".format(mask))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) # 上三角为负无穷，下三角为0.0
    print("mask2: {}".format(mask))


def test_nn_LayerNorm():

    test1 = torch.randint(0, 5, [1,2,5]).to(torch.float)  # 最后维度值为5
    normal = nn.LayerNorm(5)  # 按照最后一个维度进行规范化，这里传入5
    test2 = normal(test1)

    print("test1: {}  normal: {}".format(test1, test2))
    return

def test_deep_copy_model():
    '''
    如何深拷贝模型？两种方法
    1. copy.deepcopy
    2. type(**)model(args) 实例化   model.load_state_dict(state_dict)
    '''

    test = torch.randn([3, 2])
    model = nn.Linear(2, 2)
    test2 = model(test)
    print("test1: {} weight: {} bias: {} test2:{}".format(test, model.weight, model.bias, test2))

    # 模型拷贝的方式1
    model2 = copy.deepcopy(model)
    print("weight: {} weight2: {}".format(model.weight, model2.weight))

    # 模型拷贝方式2
    model3 = type(model)(2, 2)
    model3.load_state_dict(model.state_dict())
    
    print("weight: {} weight2: {}".format(model.weight, model3.weight))


def test_loss_funtion():

    ## 多分类的交叉熵损失
    '''
    交叉熵：它主要刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近
    H(p,q) = -sum{p(x) * log(q(x))}

    nn.CrossEntropyLoss() 等价：softmax-log-NLLLoss
    
    '''

    x_input=torch.randn(3,3)#随机生成输入  shape: 3 * 3
    print('x_input:\n',x_input) 
    y_target=torch.tensor([1,2,0])#设置输出具体值 
    print('y_target\n',y_target)

    #计算输入softmax，此时可以看到每一行加到一起结果都是1
    softmax_func=nn.Softmax(dim=1)
    soft_output=softmax_func(x_input)
    print('soft_output:\n',soft_output)

    #在softmax的基础上取log
    log_output=torch.log(soft_output)
    print('log_output:\n',log_output)

    #对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    logsoftmax_func=nn.LogSoftmax(dim=1)
    logsoftmax_output=logsoftmax_func(x_input)
    print('logsoftmax_output:\n',logsoftmax_output)

    #pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    nllloss_func=nn.NLLLoss()
    nlloss_output=nllloss_func(logsoftmax_output,y_target)
    print('nlloss_output:\n',nlloss_output)

    #直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
    crossentropyloss=nn.CrossEntropyLoss()
    crossentropyloss_output=crossentropyloss(x_input,y_target) 
    # x_input:shape是 3*3  y_target: [3] 多分类的真值(索引值,类别值)
    print('crossentropyloss_output:\n',crossentropyloss_output)


def test_data_loader():

    ## 主要看dataloader中的参数使用： collate_fn
    ## 参考：https://blog.csdn.net/dong_liuqi/article/details/114521240

    return




if __name__ == "__main__":

    test1 = torch.randn([4,1,7])

    test2 = test1[:test1.size(0), :]  #   张量切片，开辟新内存了
    print("id1: {}  id2: {}  test2: {}".format(id(test1), id(test2), test2)) ## 4 * 1 * 7  

    print(test1.shape[-1])


    # test_nn_embedding()

    # test_F_funtion()

    # test_contigous()

    # test_masked_filled()

    # test_nn_LayerNorm()

    # test_deep_copy_model()

    test_loss_funtion()
