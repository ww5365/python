# -*- encoding: utf-8 -*-
import os
import sys
import torch

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




if __name__ == "__main__":

    test_nn_embedding()
