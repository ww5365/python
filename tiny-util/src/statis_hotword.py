# -*- encoding:utf-8 -*-
import sys
import os
import re

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

class item:
    def __init__(self):
            self.name = ''     # uid or name
            self.ctr = 0.0     # ctr

if __name__ == '__main__':

    print "begin to process"
    test_file = open('ctr_test.txt')
    result = {}
    other_result = {}
    for line in test_file.readlines():
        segments = line.strip('\r\n').split('\t')
        key = "\t".join(segments[0 : 2])
        #uid = segments[2].decode('gb18030', 'ignore')
        uid = segments[2]
        #print key, uid
        #print uid
        click_cnt = float(segments[3])
        examine_cnt = float(segments[4])
        ctr = 0.0
        ctr_list = []
        ctr_list_other = []

        if examine_cnt > 0:
            ctr = click_cnt / examine_cnt
        ctr_item = item()
        ctr_item.name = uid
        ctr_item.ctr = ctr
        if not re.match(r'[+-]?\d+$', uid):
            #print key, uid
            if not result.has_key(key):
                ctr_list.append(ctr_item)
                result[key] = ctr_list
            else:
                result[key].append(ctr_item)
        else:
            if result.has_key(key):
                if not other_result.has_key(key):
                    ctr_list_other.append(ctr_item)
                    other_result[key] = ctr_list_other
                else:
                    other_result[key].append(ctr_item)

    
    ctr_hotword = 0.0
    N_hotword = 0

    ctr_poi_1 = 0.0
    N_poi_1 = 0
    ctr_poi_2 = 0.0
    N_poi_2 = 0

    for key in sorted(result):
        #print key
        value = result[key]
        for i in range(len(value)):
            ctr_hotword = ctr_hotword + value[i].ctr
            N_hotword = N_hotword + 1
            print key + "\t" + str(value[i].name) + "\t" + str(value[i].ctr)

        if other_result.has_key(key):
            ctr_list_res = sorted(other_result[key], key = lambda s: s.ctr, reverse=True)
            for i in range(len(ctr_list_res)):

                if i == 0:
                    ctr_poi_1 = ctr_poi_1 + ctr_list_res[i].ctr
                    N_poi_1 = N_poi_1 + 1
                if i == 1:
                    ctr_poi_2 = ctr_poi_2 + ctr_list_res[i].ctr
                    N_poi_2 = N_poi_2 + 1

                if i < 10:
                    print key + "\t" + ctr_list_res[i].name + "\t" + str(ctr_list_res[i].ctr)

    ctr_hotword_avg = ctr_hotword / N_hotword
    ctr_poi_1_avg = ctr_poi_1 / N_poi_1        
    ctr_poi_2_avg = ctr_poi_2 / N_poi_2        

    print ctr_hotword_avg,ctr_poi_1_avg,ctr_poi_2_avg



