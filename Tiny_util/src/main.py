# -*- encoding: utf-8 -*-
'''
Created on Aug 22, 2018

@author: wangwei69
'''

import re

def is_chinese(u_str):

    """判断一个unicode字符串是否为纯中文"""
    for ch in u_str:
        if ch >= u'\u4e00' and ch<=u'\u9fa5':
            continue
        else:
            return False
        
    return True   

# 纯中文判断: 中文 数字
def judge_pure_chinese(keyword):  
    return all((u'\u4e00' <=c<= u'\u9fff'  or u'\u0030' <= c <= u'\u0039') for c in keyword)

'''
判断是纯英文的str
'''

def judge_pure_english(keyword):  
    return all(ord(c) < 128 for c in keyword) 


def lcs():
    print("enter into lcs!") 
    
def str_use_test():
    row = "王伟-地铁 你好"
    wds = row.split()
    print("wds:", wds)
    query_list = []
    
    wds_len = len(wds)
    if wds_len == 1:
        query_list.append(wds[0])
    if wds_len >=2:
        
        if is_chinese(wds[wds_len-1].decode("gb18030")):
            print("all chinese last word")
            for i in range(len(wds) - 1):
                query_list.append(wds[i])
        else:
            print("not all chinese word")
            for i in range(len(wds)):
                query_list.append(wds[i])        
     
    query = ' '.join(query_list)      
    query_new = query;
    query = query.decode("gb18030")
    indx = query.find("-地铁".decode("gb18030"))
    indx2 = query.find("-公交".decode("gb18030"))
    if query.find("(".decode("gb18030")) >-1 or query.find("-".decode("gb18030")) >-1:
        print("find zifu:.......",query)
        
    if indx >=0:
        query_new = query[0:indx]
    
    if indx2 >=0:
        query_new = query[0:indx2]    
        
    print("query_new:", query_new)
    print("query_new:", query_new.encode("gb18030"))
    
    if row.isalpha() == True:
        print("row isalpha", row)
    else:
        print("row not alpha", row)
    
    ##特殊字符通过正则来进行匹配
    query_test2 = "beiji:ng?。ni-ah"
    charset = "[\.\"\'?()@&#\-,:。，]+"
    pattern = re.compile(charset.decode("gb18030"))
    delete_pos = pattern.findall(query_test2.decode("gb18030"))    
    
  
    print("delete_pos:", query, delete_pos)
    
    query_test3 = "王伟你3333 好"
    res = judge_pure_chinese(query_test3.decode("gb18030"))
    
    query_test4 = "王伟"
    print " ".join(["test 截取",query_test3[0:len(query_test4)]])
    
    if query_test4 == query_test3[0:len(query_test4)]:
        print "good 截取成功"
    else:
        print "bad 截取失败"
    
    



        
        

def list_use_test():
    
    gen_list = list();
    gen_list.append("kv")
    gen_list.append("我们")
    
    print("gen_list", gen_list)    
    
    if "kv" in gen_list:
        print("kv find!!!")
    
def dict_use_test():
    
    #获取二维dict中所有的key值
    
    poi=dict()
    for i in range(5):
        poi[i] = dict()
        poi[i]["tkey"] = 1
    
    for key in poi.keys():
        for key2 in poi[key]:
            print(key2)  
            
    if 2 in poi.keys():
        print "has key judge:"
        print poi[2]            
    
    
def main():
    print ("enter into main!")
    lcs()
    str_use_test()
    list_use_test()
    dict_use_test()
    
    keyword = "wang nihao" 
    res = judge_pure_english(keyword)
    print res
    print(keyword.upper())
    
    #vec使用
    
    examine_vec = [0 for j in range(10)]
    
    
    print "\t".join(map(str,examine_vec))
    
    keyword2 = "wei"
    
    print(keyword, keyword2)
    
    loc = "123,345"
    print loc[1:-1]
    print loc[:-1]
    
    loc_pair = loc[1:-1].split(",")
    
    if len(loc_pair) == 2:
        (x,y) = loc_pair
        
    print "loc: x=%s y=%s" %(x, y)   
    
    x=None
    if x is None:
        print "x is None"
        
    print  range(10-2)
    
    for i in range(8):
        print i
        
    action = [0 for i in range(12)] 
    
    action = [1,2,3,4,5,6,7,8,9,10]
    print action[3:5]
    
    

if __name__ == '__main__':
    main()
    
    info = []
    info0 = ["test1","test2"]
    info1 = ["test3","test2"]
    info.append(info0)
    info.append(info1)    
    print info
    
    
    
    