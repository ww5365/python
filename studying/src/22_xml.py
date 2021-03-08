# -*- coding: utf-8 -*-

import xml.etree.ElementTree as et
from xml.dom.minidom import parseString
import dicttoxml
import os
import codecs

cur_path = os.path.dirname(__file__)
cur_file = os.path.join(cur_path, 'data\\test.xml')
print(cur_file)

'''
Python 标准库 有三种方法解析 XML：SAX，DOM，以及 ElementTree。
1. SAX：来自Python 标准库，SAX 用事件驱动模型，通过在解析XML的过程中触发一个个的事件并调用用户定义的回调函数来处理XML文件。
import xml.sax
2. DOM：是W3C组织推荐的处理可扩展置标语言的标准编程接口，一个 DOM 的解析器在解析一个 XML 文档时，一次性读取整个文档，把文档中所有元素保存在内存中的一个树结构里，之后你可以利用DOM 提供的不同的函数来读取或修改文档的内容和结构，也可以把修改过的内容写入xml文件。
from xml.dom.minidom import parse
3. ElementTree：ElementTree就像一个轻量级的DOM，具有方便友好的API。代码可用性好，速度快，消耗内存少。
from xml.etree.ElementTree import ElementTree

ps:其他第三方工具包有：
untangle工具包，将xml格式转换成python object；
xmltodict工具包，像操作dict一样来操作xml  pip install xmltodict

ref: https://blog.csdn.net/guangmingsky/article/details/85873069

'''


def dictxml(tag, dict_value):
    '''
    turn a dict to xml
    '''

    elem = asElement(tag)


def test_dict2xml():
    # 都是属性值，怎么表示属性
    # dict_value = {
    #    'PPML': {
    #        'version': '3.0',
    #        'xmlns': 'http://www.dmg.org/PMML-3-0',
    #        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema_instance',
    #        'labels': {
    #            'label': {
    #                'properties': {
    #                    'property':}
    #            }
    #        }
    #    }
    # }
    '''
    方法1：
    xml.etree.ElementTree as et

    '''
    dict_value = {
        'PPML': {
            'version': '3.0',
            'b': 'bvalue',
            'a': 'avalue'
        }
    }

    ppml = et.Element('PPML')

    for key, value in dict_value.items():
        labels = et.SubElement(ppml, 'ppml')
        # dict 操作，如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值
        value.setdefault('author', 'ww')

        for key2, value2 in value.items():
            lable = et.SubElement(labels, key2)
            lable.text = str(value2)

    xml = et.tostring(ppml)   # 利用et把形成的xml转成raw string
    print("-------raw xml---------")
    print(xml)

    # 利用xml.dom.minidom 把raw xml 修改后，转成格式化的xml
    print("-------pretty xml---------")
    dom = parseString(xml)
    print(dom.toprettyxml(' '))  # 使用空格进行缩进

    '''
    方法2：
    使用dict2xml模块进行转换
    ref: https://cloud.tencent.com/developer/article/1737945
    '''
    d = [20, 'name',
         {'name': 'apple', 'num': 10, 'price': 23},
         {'name': 'pear', 'num': 20, 'price': 18.7},
         {'name': 'banana', 'num': 10.5, 'price': 23}]

    bxml = dicttoxml.dicttoxml(d, custom_root='fruit')
    xml = bxml.decode('utf-8')

    print("-------dicttoxml xml----")
    print(xml)

    dom = parseString(xml)
    pxml = dom.toprettyxml(indent=' ')

    # 把pretty xml 写入文件
    f = codecs.open(cur_file, 'w', encoding='utf-8')
    f.write(pxml + "\n")
    f.close()


if __name__ == "__main__":

    test_dict2xml()
