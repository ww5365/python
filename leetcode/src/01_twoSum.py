# -*- coding: utf-8 -*-
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        '''
        test for wirte code
        '''
        for i in range(len(nums)):
            print(nums[i])
        return target
            
if __name__ == '__main__':
    
    s = Solution()
    li = [1,2,5,4]
    res = s.twoSum(nums = li, target = 5)
    print("res: ", res)
    