
from  typing import List
import os


class Solution:
    
    def __init__(self):
        pass

    def find_left(self, height: List[int], idx: int) ->int:
        
        if idx <= 0:
            return 0
        i = idx - 2
        max_height = height[idx - 1]
        res = 1
        while i >= 0:
            if max_height < height[i]:
                res += 1
                max_height = height[i]
            i = i - 1  
                
        return res
    
    def find_right(self, height: List[int], idx: int) -> int:
        if idx >= len(height)- 1:
            return 0
        max_height = height[idx + 1]
        res = 1
        i = idx + 2
        while i <= len(height) - 1:
            if max_height < height[i]:
                res += 1
                max_height = height[i]
            i = i + 1
        
        return res
    
    def number_of_mountain_seen(self, height: List[int]) -> int:
        if len(height) <= 2:
            return 0
        res = 0
        left_last_res = 1
        right_last_res = 1
        
        for i in range(len(height)):
            
            if i != 0 and (i+1) % 2 == 1:
                if height[i - 1] < height[i - 2]:
                    res += left_last_res + 1
                    left_last_res += 1
                elif height[i - 1] == height [i - 2]:
                    res += left_last_res
                else :    
                    res += self.find_left(height, i)
            elif i < (len(height) - 2) and (i + 1)% 2 == 0:
                if height[i+1] < height[i+2]: 
                    res += right_last_res + 1
                    right_last_res += 1
                elif height[i+1] == height[i+1]:
                    res += right_last_res
                else:
                    res += self.find_right(height, i)
            elif i == (len(height) - 2) and (i + 1)% 2 == 0:
                res += 1
        return res
    
    
if __name__ == '__main__':
        
        s = Solution()
        li = [16,5,3,10,21,7]
        res = s.number_of_mountain_seen(li)
        
        print(res)
        