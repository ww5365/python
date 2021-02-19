# -*- encoding: utf-8 -*-


if __name__ == "__main__":

    b = 0b0101  # 0b  二进制
    c = 0o17  # 0o  八进制
    d = 0x1e  # 0x 十六进制

    print("b,c,d", b, c, d)

    # 十进制转成二进制，八进制，十六进制
    a = 20
    print("binary: ", bin(a))
    print("oct: ", oct(a))
    print("hex: ", hex(a))

    # 类型转换：int->str  str->int bool->int int->float
    # age = input("how old are you?\n")
    age = '35'
    print("my age is :", age)
    print("next year my age is:", int(age) + 1)

    m = '1a2e'  # 把字符串(按照特定进制)转成整数
    print(int(m, 16))

    # 基本运算 ： + -  * / //（整除） %

    print((9**2))  # 9^2 = 81
    print(81 ** (1/2))  # 81^0.5 = 9
    print(10//3)  # 整除
    print(10 % 3)  # 求模

    #  赋值运算符
    m, n = 8, 9   # 拆包
    print(m, n)
    o, *p, q = 1, 2, 3, 4, 5  # 可变长度变量,p=[2,3,4]
    print(o, p, q)

    # 位运算符
    color = 0x23e48f
    red = color & 0x0000ff
    print(hex(red))
