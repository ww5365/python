import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,180,180)  # 从0到180 画180个点

'''
画分段函数
'''

#画强时效
interval0 = [1 if (i<=30) else 0 for i in x]
interval1 = [1 if (i>30) else 0 for i in x]

# 画分段函数
interval0 = [1 if (i<=14) else 0 for i in x]
interval1 = [1 if ((i>14) and i <= 30) else 0 for i in x]
interval2 = [1 if (i>30) else 0 for i in x]
# y = np.exp(x)* interval0 + x * interval1 + np.sin(x)*interval2
#y = 1/(1+np.exp(-0.01*(360-x))) - 0.5  # sttime=360  shrinking=60  prob=0.6 plt.title("stTime:360, shrinking:60 ,prob:0.6")
# y = (interval0 * 1/(1+np.exp(-0.04*(30-x))) + interval1 * 1/(1+np.exp(-0.01*(30-x)))) - 0.5  #plt.title("stTime:30, shrinking:x<30 = 15 x>30 = 60, prob:0.6") 
# plt.title("stTime:30, shrinking: x<30->15 x>=30->60, prob:0.6") 

# y = (interval0 * 1/(1+np.exp(-0.8*(1/4)*(14-x))) + interval1 * x/x * 0.5 + interval2 * 1/(1+np.exp(-0.01*(30-x)))) - 0.5  
# plt.title("stTime:14, shrinking: x<14->4 x>=30->60, prob:0.8")

# plt.xlabel("distanceDays : days of between (currentTime,docTime)")
# plt.ylabel("weight : negative downgrade, positive weighted")

# plt.plot(x,y)
# plt.show()

# print(1/(1 + np.exp(-1))-0.5)


'''
同一个图上画多个曲线, 同时每个曲线是分段的函数
'''

# fig, ax = plt.subplots() # 创建图实例
# # x = np.linspace(0,2,100) # 创建x的取值范围
# boostFactor0 = (1/(1+np.exp(-0.8*(1/4)*(14-x))) - 0.5)/2
# boostFactor2 = (1/(1+np.exp(-0.8*(1/60)*(30-x))) - 0.5)/15

# y1 =  1 + ((interval0 * boostFactor0) + (interval1 * x/x * 0)  + interval2 * boostFactor2)
# ax.plot(x, y1, label='good revelance: boostfactor>0 scale=2, boostfactor<0 scale=15') # 作y1 = x 图，并标记此线名为linear

# boostFactor00 = (1/(1+np.exp(-0.8*(1/4)*(14-x))) - 0.5)/4
# boostFactor22 = (1/(1+np.exp(-0.8*(1/60)*(30-x))) - 0.5)/7.5
# y2 =  1 + ((interval0 * boostFactor00) + (interval1 * x/x * 0)  + interval2 * boostFactor22)
# ax.plot(x, y2, label='revelance: boostfactor>0 scale=4, boostfactor<0 scale=7.5') #作y2 = x^2 图，并标记此线名为quadratic

# boostFactor000 = (1/(1+np.exp(-0.8*(1/4)*(14-x))) - 0.5)/20
# boostFactor222 = (1/(1+np.exp(-0.8*(1/60)*(30-x))) - 0.5)/6
# y3 =  1 + ((interval0 * boostFactor000) + (interval1 * x/x * 0)  + interval2 * boostFactor222)
# ax.plot(x, y3, label='weak revelance: boostfactor>0 scale=20, boostfactor<0 scale=6') # 作y3 = x（^3 图，并标记此线名为cubic


# ax.set_xlabel('distanceDays : days of between (currentTime,docTime)') #设置x轴名称 x label
# ax.set_ylabel('boostScore') #设置y轴名称 y label
# ax.set_title('the boost weight of docs based on timeliness and relevance') #设置图名为Simple Plot
# ax.legend() #自动检测要在图例中显示的元素，并且显示

# plt.show() #图形可视化



'''
一张图上画两个坐标轴，每个坐标轴可以展示不同的画图方式：
图1连续函数线
图2坐标点的散点图
使用相同的y轴
'''

# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
print(np.pi)
y = np.sin(x**2)

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  # 共享y轴
ax1.plot(x, y)
ax1.set_title('Sharing Y axis') 
ax2.scatter(x, y)  # 画坐标点图
ax2.grid()  # 右图坐标中显示网格

plt.show()





