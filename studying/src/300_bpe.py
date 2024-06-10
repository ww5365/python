# -*-encoding: utf-8 -*-

'''
BPE : byte pairs encoding  字节对编码

参考：https://zhuanlan.zhihu.com/p/448147465

'''

import re, collections

def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    # print("******************\n{} {}".format(bigram, p))
    for word in v_in:
        w_out = p.sub(''.join(pair), word)  # w e s t </w> => w es t </w>

        # print("******************\n{} {}".format(word, w_out))
        
        v_out[w_out] = v_in[word]
    
    print("v_in : {}  v_out: {}".format(v_in, v_out))
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens


# 构建下面单词的词典

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
print('==========')
print('Tokens Before BPE')
tokens = get_tokens(vocab)
print('Tokens: {}'.format(tokens))
print('Number of tokens: {}'.format(len(tokens)))
print('==========')

num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)

    print("pairs: {}".format(pairs))
    if not pairs:
        break
    best = max(pairs, key=pairs.get)  # key= ?  获取字典中最大value对应的key值  当max() 函数中有 key 参数时，求的是 value 的最大值，当没有 key 参数时，求的是 key 的最大值。
    vocab = merge_vocab(best, vocab)
    print('Iter: {}'.format(i))
    print('Best pair: {}'.format(best))
    print('Vocab : {}'.format(vocab))
    tokens = get_tokens(vocab)
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))   
    print('=======================================')


