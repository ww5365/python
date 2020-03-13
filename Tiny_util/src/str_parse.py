#--*-- encoding:utf8 --*--

'''
Created on Dec 18, 2018

@author: wangwei69
'''

import json

def str_to_dict():
    
    str = "index$1\twangwei$nihao" 
    
    str_arr = str.split("\t")
    
    kv_dict = {}
    
    for kv in str_arr:
        kv = kv.split("$")
        kv_dict[kv[0]] = kv[1]
        
    kv_dict_str = json.dumps(kv_dict)    
    
    print "%s" % kv_dict_str


def str_encode_language():
    
    '''
    query 类型：
    纯中文
    纯英文、拼音
    中文+英文
    中文+数字
    英文+数字
    
    '''
    
    #info: nihao王伟
    json_str = '{"name":"wangwei", "id":"王12w@*+ a", "info":"\u006e\u0069\u0068\u0061\u006f\u738b\u4f1f"}'
    
    str = json.loads(json_str) # 每个key对应的value都是unicode编码，所以id的长度是：9 
    
    '''
    不同编码情况下，中英文混合的字符串，长度是不同的；
    unicode：7 每个字母或汉字，算1个"字符"。最适合for循环处理了；每个"字符"实际占用2个字节。
    utf8：11  字节数 5+6 = 11
    gb18030: 9 字节数 5+4 = 9
    
    '''
    print str["info"], len(str["info"]);
    print str["info"].encode("utf-8"), len(str["info"].encode("utf-8"));
    print str["info"].encode("gb18030"), len(str["info"].encode("gb18030"));

    #使用unicode编码方式：中文，英文，特殊字符，都是长度：1
    for i in range(len(str["id"])):
        #str["id"][i] 对应是unicode编码；再转成utf8编码来判断是否是英文字符
        key = str["id"][i].encode("utf8")
        if key.isalpha():
            print "alpha:%s" %(key) #英文字符，是alpha
        else:
            print "not alpha:%s" %(key) #中文，标点符号，特殊字符都不是alpha
    
    str2 = "王12w@*+ a"  #str2保持和源码一致的编码方式：utf8，所以长度是：11
    
    for i in range(len(str2)):
        key2 = str2[i]
        if key2.isalpha():
            print "alpha:%s" %(key2)
        else:
            print "not alpha:%s" %(key2)
    
    str3 = "what is"
    print str3.isalpha() #不是alpha 因为存在一个空格
    str4 = "ab测试" 
    print str4.isalpha() #不是alpha 有中文
    str5 = "ab123"
    print str5.isalpha() #不是aplpha 有数字字符
    
    
    
    


if __name__ == "__main__":
    
    print "str parse!"
    str_to_dict()
    str_encode_language()
    
    
    
    