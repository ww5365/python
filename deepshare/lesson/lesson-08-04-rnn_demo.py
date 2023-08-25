# -*- coding: utf-8 -*-
"""
# @file name  : rnn_demo.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2019-12-09
# @brief      : rnn人名分类
"""
from io import open
import glob
import unicodedata
import string
import math
import os
import time
import torch.nn as nn
import torch
import random
import matplotlib.pyplot as plt
import torch.utils.data
import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__) + os.path.sep + "..")
sys.path.append(hello_pytorch_DIR)

from tools.common_tools import set_seed

set_seed(1)  # 设置随机种子
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)   # 把一串UNICODE字符串转换为普通格式的字符串，具体格式支持NFC、NFKC、NFD和NFKD格式
        if unicodedata.category(c) != 'Mn'  # Mark, Spacing Combining
        and c in all_letters)

'''
unicode ： utf-8  变长，实现unicode
U+ 0000 ~ U+ 007F: 0XXXXXXX
U+ 0080 ~ U+ 07FF: 110XXXXX 10XXXXXX
U+ 0800 ~ U+ FFFF: 1110XXXX 10XXXXXX 10XXXXXX
U+10000 ~ U+1FFFF: 11110XXX 10XXXXXX 10XXXXXX 10XXXXXX

'''

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)  # 返回两个tensor，(values=tensor([4.]),indices=tensor([3])) 第1个是最大值的张量，第2个是最大值下标的张量
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)                 # 选类别
    line = randomChoice(category_lines[category])           # 选一个样本
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)    # str to one-hot
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))


def get_lr(iter, learning_rate):
    lr_iter = learning_rate if iter < n_iters else learning_rate*0.1
    return lr_iter

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.u = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)  # 先求softmax 再求了log

    def forward(self, inputs, hidden):

        u_x = self.u(inputs)  # U*X

        hidden = self.w(hidden) # W*S
        hidden = self.tanh(hidden + u_x) # f(U*X + W*S)

        output = self.softmax(self.v(hidden)) # softmax

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    line_tensor = line_tensor.to(device)
    hidden = hidden.to(device)
    category_tensor = category_tensor.to(device)

    for i in range(line_tensor.size()[0]):  # 维度：字符串长度; 每个字母是one-hot表征;
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


if __name__ == "__main__":
    # config
    data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "rnn_data", "names"))
    if not os.path.exists(data_dir):
        raise Exception("\n{} 不存在，请下载 08-05-数据-20200724.zip  放到\n{}  下，并解压即可".format(
            data_dir, os.path.dirname(data_dir)))

    path_txt = os.path.join(data_dir, "*.txt")
    
    # main中声明的变量，默认是global的 
    all_letters = string.ascii_letters + " .,;'"     #ascii_letters是生成所有字母，从a-z和A-Z, digits是生成所有数字0-9.
    n_letters = len(all_letters)    # 52 + 5 字符总数
    # print_every = 5000
    # plot_every = 5000
    print_every = 1
    plot_every = 1
    learning_rate = 0.005
    # n_iters = 200000
    n_iters = 1

    # step 1 data
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    for filename in glob.glob(path_txt):  # python自带的文件操作模块,glob.glob(PATH), 主要支持文件名的通配符, 返回文件路径list(仅当前目录下)
        category = os.path.splitext(os.path.basename(filename))[0]  # 把Arabic.txt中的Arabic提取出来    
        all_categories.append(category)   # 语言类型，可以任务是label
        lines = readLines(filename)
        category_lines[category] = lines

        # if category == "Arabic":
        #     print("category: {} : {}".format(category, lines))

    n_categories = len(all_categories)


    # step 2 model
    n_hidden = 128
    # rnn = RNN(n_letters, n_hidden, n_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)  # 57， 128， 语种种类数

    rnn.to(device)

    # step 3 loss
    # 负对数似然损失函数（Negative Log Likelihood）
    criterion = nn.NLLLoss()  # nn.LogSoftmax(dim = 1) 取绝对值，再取对应label索引位置上的值，拿出来求平均值。

    # step 4 optimize by hand

    # step 5 iteration
    current_loss = 0
    all_losses = []
    start = time.time()
    for iter in range(1, n_iters + 1):
        # sample  从某个语种中选择某个姓名, 语种类别向量category_tensor: 正确label标签  line_tensor: 某个姓名的one-hot表示
        category, line, category_tensor, line_tensor = randomTrainingExample()

        # line_tensor : <line_length x 1 x n_letters>  one-hot表示
        # category_tensor: [4] 某个语种的下标索引构成的张量
        print("category: {} line:{} category_tensor:{} line_tensor:{}".format(category, line, category_tensor, line_tensor ))

        # training
        output, loss = train(category_tensor, line_tensor)

        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('Iter: {:<7} time: {:>8s} loss: {:.4f} name: {:>10s}  pred: {:>8s} label: {:>8s}'.format(
                iter, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

path_model = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "rnn_state_dict.pkl"))
# if not os.path.exists(path_model):
#     raise Exception("\n{} 不存在，请下载 08-05-数据-20200724.zip  放到\n{}  下，并解压即可".format(
#         path_model, os.path.dirname(path_model)))
torch.save(rnn.state_dict(), path_model)
plt.plot(all_losses)
plt.show()

predict('Yue Tingsong')
predict('Yue tingsong')
predict('yutingsong')

predict('test your name')



