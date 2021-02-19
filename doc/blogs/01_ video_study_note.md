#  qianfeng python 2020 教程

[参考视频](https://www.bilibili.com/video/BV1qK411n7gQ?p=20)


## day 1


### 基本数据类型
1、进制
```
b = 0b0101  #二进制
c = 0o17 # 八进制
d = 0x1E # 十六进制
```

2、基本数据类型转换
3、运算符
* 算数运算符
* 字符串中运算符： + 
* 赋值： = += -= *= /= **=  可变长度赋值 *var
* 比较运算
* 逻辑运算： and or not
* 位运算： & | ^ ~取反  <<左移  >>右移


4、接受用户的基本输入: input(prompt) -> str  

```
input_str = input("input your info: ")
```

### 字符串

 #### 字符串表示

 * 三个单引号：多行字符串, 所见所得
 * raw string： 使用r
 ```
 str = "wangwei"  # 注意，在此字符串上的操作，不会更改字符串内容
 str2 = r"wangwei\t\n" #不会转义
 str3 = '''
    create table
    clolum1,
    clolum2,
 '''

 ``` 

 ####  字符串操作

 * 切片：
 * 分割: split, partition
 * 