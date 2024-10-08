# -*- encoding: utf-8 -*-

import os
import re
'''
re： 正则表达式模块的使用

重要参考：
https://www.runoob.com/python3/python3-reg-expressions.html


特殊字符类：
.	匹配除 "\n" 之外的任何单个字符。要匹配包括 '\n' 在内的任何字符，请使用象 '[.\n]' 的模式。
\d	匹配一个数字字符。等价于 [0-9]。
\D	匹配一个非数字字符。等价于 [^0-9]。
\s	匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。
\S	匹配任何非空白字符。等价于 [^ \f\n\r\t\v]。
\w	匹配包括下划线的任何单词字符。等价于'[A-Za-z0-9_]'。
\W	匹配任何非单词字符。等价于 '[^A-Za-z0-9_]'。

正则表达式[\w]+,\w+,[\w+] 三者有何区别：
    [\w]+和\w+没有区别，都是匹配数字和字母下划线的多个字符；
    [\w+]表示匹配数字、字母、下划线和加号本身字符；

匹配字符个数：
    用*表示任意个字符（包括0个）
    用+表示至少一个字符
    用?表示0个或1个字符
    用{n}表示n个字符
    用{n,m}表示n-m个字符：
    星号 (*) 匹配前面的字符 0 次或多次。
    .* : 匹配任意字符

    例如，10* 可以匹配：
    1
    10
    100
    1000


    问号 (?) 匹配前面的字符 0 次或 1 次。
        例如，10? 可以匹配：
        1
        10

分组：()
(abc|bcd|cde)，表示这一段是abc、bcd、cde三者之一均可，顺序也必须一致

'''


def test_re_match():
    '''
    re.match(pattern, string, flags=0):

    功能：从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回None
    参数说明：
    flags: re.I 大小写不敏感  re.M 多行匹配   多个标识使用： |  来生效
    输出：
    匹配对象或None
    group(num = 0):
      匹配的整个表达式的字符串，group() 可以输入多个组号，它返回包含那些组所对应值的元组

    groups():
        返回一个包含所有小组字符串的元组，从 1 到 所含的小组号

    另外：span函数用来返回匹配对象起始位置和结束位置

    '''

    print(re.match('wwW', 'www.baidu.com www.huawei.com',
                   re.I).span())  # re.I 忽略大小写；span（）返回[begin, end) 结束位置 (0, 3)

    print(re.match('wwW', 'www.baidu.com www.huawei.com',
                   re.I).groups())  # groups(): 空元祖，因为是从1开始的

    print(re.match('bai', 'www.baidu.com', re.I))  # 不在起始位置匹配，返回None

    line = 'Cats are smarter than dogs'

    #  .* 匹配任意字符，至少0个字符 () 是1个匹配组  .+ 匹配任意字符，至少1个字符
    # (.*?) 是非贪婪匹配,加上? 以最少的可能性匹配 默认是贪婪方式，尽可能多的可能来匹配
    match_obj = re.match(r'(.*) are (.+?)(.+)', line, re.I)

    if match_obj:
        print(match_obj.groups())  # 返回一个包含所有小组字符串的元组
        print(match_obj.group())  # 匹配的整个表达式的字符串结果
        print(match_obj.group(1))  # 从1开始，匹配对象中的索引
        print(match_obj.group(2))
        # print(match_obj.group(3))
    else:
        print("match result is None!")


def test_re_search():
    '''
    re.search(pattern, string, flags = 0)

    输入和输出：类似re.match

    区别： 可以在任意位置开始匹配，匹配到就返回匹配对象，否则None

    '''
    print(re.search('BAI', 'www.baidu.com', re.I).span())  # 从bai开始匹配成功
    str1 = 'A23G4HFD567'
    # (?P<name> Expression) 命名捕获组 可以通过name来使用匹配组的值
    str_match = re.search(r'(?P<value>\d+)', str1)
    print("re.search :  ", str_match.group('value'))


def test_re_sub():
    '''
    re.sub(pattern, repl, string, count=0, flags=0):
    功能：匹配到特定模式的字符串，并将匹配部分替换为指定的repl
    输出：处理处理后的字符串

    输入：
    pattern : 正则中的模式字符串。
    repl : 替换的字符串，也可为一个函数。
    string : 要被查找替换的原始字符串。
    count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
    flags : 编译时用的匹配模式，数字形式。
    前三个为必选参数，后两个为可选参数。

    '''
    print("--" * 30)
    phone = "010-1891234567 #电话号码"
    # phone = ["010-1891234567 #电话号码", "192002212"]
    # 删除#开头的注释
    phone = re.sub(r'#.*$', '', phone)
    print(phone)
    # 去掉非数字的字符
    phone = re.sub(r'\D', '', phone)
    print(phone)

    line = r"#E-s[傲慢]-表情"

    print("before sub:", line)
    # 正则匹配串里的()是为了提取整个正则串中符合括号里的正则的内容
    line = re.sub(r"#E\-[\w]*\[(得意|傲慢)+\]", "α", line)
    print("sub:", line)


def string_filter(input_str: str):
    # 使用[]限定过滤集合
    pat = r"[`~!@#$%^()={}:;\[\]<>《》/！￥…（）【】‘；：”“\"’。、?·,，]"
    if len(input_str) <= 0:
        return input_str
    else:
        # print("before: ", input_str)
        input_str = re.sub(pat, ' ', input_str).strip()  # 把pat集合中字符，匹配到的换成空格
        # print("after: ", input_str)
        input_str = re.sub(r"\.{2,}", " ", input_str)  # 把连续2个以上的.替换成1个
        # print("after2: ", input_str)
        input_str = re.sub(r" {2,}", " ", input_str)  # 把连续2个以上的空格替换成1个
        # print("afte3: ", input_str)
    return input_str


if __name__ == '__main__':

    test_re_match()
    test_re_search()
    test_re_sub()

    # 想把中英文的逗号过滤掉
    str2 = 'Comuna ，Etropole, Sofia'
    print(string_filter(str2))

    # 把以_std结尾的字符串替换为空
    str3 = "test_for_std"
    print(re.sub(r"_std$", "", str3))

    # 把以_aliasN结尾的替换成空
    key = "zh_hk_alias"
    key = re.sub(r'_alias\d*', '', key)
    print(key)
