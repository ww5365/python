import math
import os
from typing import List

class Solution:
    
    '''
    房间除湿
    '''
    
    def is_finished(self, humidity):
        for i in len(humidity):
            if humidity[i] != 0:
                return False
        return Ture
    
    def find_min_dis(self, humidity, single, multiple):
        single_dis = 10000000
        multiple_dis = 10000000
        single_pos = -1
        multiple_pos = -1
        for i in range(len(humidity)):
            if  humidity[i] > 0  and single_dis > math.fabs(humidity[i] - single):
                single_pos  = i
                single_dis = math.fabs(humidity[i] - single)
            if  humidity[i] > 0  and multi_dis > math.fabs(humidity[i] - multiple):
                multiple_pos  = i
                multiple_dis = math.fabs(humidity[i] - multiple)
        return  single_pos, multiple_pos
    
            
    def num_cycles(self, humidity: List[int], single: int, multiple: int) -> int:
        
        res = 0
        
        while !(is_finished(humidity)):
            res += 1
            single_pos, multiple_pos = self.find_min_dis(humidity, single, multiple)
            
            if single > multiple:
                pos = self.find_min_dis(humidity, single)
                humidity[pos] -= single
                for i in range(len(humidity)):
                    if i != pos:
                        humidity[i] -= multiple
            else:
                
 
                
                
                
            
                
        
        
        
        