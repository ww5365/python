# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from typing import List, Dict

'''
题目：
https://leetcode-cn.com/problems/container-with-most-water/

Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0). 
Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.

指针移动规则与证明:?

'''


class Solution:
    def __init__(self, a: int = 0):
        self.a = a

    def maxArea(self, height: List[int]) -> int:

        i, j, res = 0, len(height)-1, 0
        while i < j:
            if height[i] < height[j]:
                res = max(res, height[i] * (j - i))
                i += 1
            else:
                res = max(res, height[j] * (j - i))
                j -= 1

        return res


if __name__ == '__main__':

    arr1 = np.random.randint(1, 10, size=(1, 6))  # 生成了1*6的二维数组
    print(arr1[0].tolist())
    s = Solution()
    li = arr1[0].tolist()
    res = s.maxArea(li)
    print(res)
