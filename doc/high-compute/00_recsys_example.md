
# 

## feature_id space

推荐平台的feature_id space 的理解？

feature_id space 不是特征个数，是特征取值的总数。

比如：我有两个特征： 爱好和性别， 爱好取值：20个  性别取值：2 个  那么我的feature_id space是20+2=22.
也就是我的token 总数是22个。 

| 术语               | 含义       |
|------------------|----------|
| feature          | 字段       |
| feature value    | token    |
| feature_id       | token 编号 |
| feature_id space | token总数  |



##  dynamic embedding

推荐系统，工作界面临的大难题：embedding的规模   feature_id space巨大，假设为10亿。单张embedding table大小：
100亿*128维*4byte 约为 5TB  GPU的显存根本装不下。


传统解决方法：

- parameter server   
- cpu embedding   : 问题？gpu访问cpu embedding 通过PCIe 延迟非常高
- ssd embedding

出现了dynamic embedding(动态embedding缓存) 方案，核心思想：
* 只把热点 embedding 放到gpu   : L1 cache
* 冷数据放cpu/ssd             : L2 storage



