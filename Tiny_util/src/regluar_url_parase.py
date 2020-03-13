# --*-- encoding: utf-8 --*--
'''
Created on Dec 3, 2018
@author: wangwei69
'''

import os
import re
import sys

def remove_color(poi_name):
    
    poi_name = poi_name.replace("<s_0>", "");
    
    return;

def parse_poi_from_string(json_str):
    json_str_array = json_str.split("s\":")
    json_str = json_str_array[1]
    m = re.findall(r"\"(.*?)\"\<(\w*)\>", json_str)
    
    poi_str = ""
    if m != "[]":
        for i in range(0, len(m)):
            detail = remove_color(m[i][0])
            detail = eval("'%s'" % detail)
            uid = m[i][1]
            poi_str += "," + detail + "||" + uid
            
    return poi_str.strip(",")


def regular_use():
    

    
    line = 'NOTICE: 11-19 23:59:02:  sug-as * 39350 123.123.249.221 - - [19/Nov/2018:23:59:02 +0800] "GET /mapssu?st=0&dpi=(489,489)&abtest=16140,16221,16257&cid=1&sign=ecb64930e7da2260dabac30bb37cbe39&zid=Z1Ub1iFVAr1hAIdWCRM1d8-eKV1eLg28A4uZ7ibP-iuleLIl2hFX38cyr4jddc1vRzGJtr34lJwZ810enejHmNA&resid=01&net=1&wd=s&cuid=4bf666b855da7fc02cb8c5223208fdad&sinan=vXrq2V25jP684mQ__ubr%2BOAHR&sv=10.11.0&gid=G1YM7B338FQYCVSLZBOOKZLTL6CUYJ65K5&l=7&screen=(1125,2436)&ver=1&b=(15389112,201306%3B16631990,2892551)&ctm=1542643142.016000&channel=1008648b&highlight_flag=2&ndid=fbpKdv5rpfqMki7K4fdayw5_EzTFnWXdDLwDNbKAic7cZvX68iUCL2_zb&rp_format=pb&co=460%3A01&os=iphone12.100000&oem=&bduid=RCNhKSIOR&type=0&cpu=ARMv7&loc=(12950552.365182,4837683.418343)&mb=iPhone10,3&qt=sug" 0 0 mod_gzip:0pct. "" "" "" PREDICT_QUERY=[] ORI_QUERY=[s] QUERY=[s] SU [{"q":"s","p":true,"t":0,"s":["松发肉骨茶(克拉码头店)$新加坡$20001$2276307f470b0e033210af07$"<703736361>,"新加坡<s_0>s<\/s_0>.e.a.海洋馆$新加坡$20001$dc5d08556a82e35a970319a4$"<180267759>,"<s_0>s<\/s_0>imon star show phuket$普吉$20553$a3f8970b996927a041ab9a2d$"<834538608>,"<s_0>s<\/s_0>ingapore zoological gardens$新加坡$20001$d4540fac7af2be0baf080f71$"<210597422>,"夜间野生动物园$新加坡$20001$c3bdfd5bff003ccb5c17c35f$"<180265417>,"拉威海鲜市场$普吉$20553$828b7b6703a5bc5cbd4842cb$"<842524367>,"圣安德烈教堂$新加坡$20001$87ac93de55192ee8dbd89190$"<703182668>,"<s_0>s<\/s_0>taroměstské nám.$布拉格$46663$35b62c0dd240c0642002627d$"<290291097>,"<s_0>s<\/s_0>ydney opera house$悉尼$31765$6336282eba85655ecbe33c43$"<694708000>,"<s_0>s<\/s_0>ingapore botanic gardens$新加坡$20001$7770b5b349e09ae0132935a4$"<180267731>]}] err=0,timeused 70,sug_as_time=36,da_time=0,as_time=0,bdrp_time=0,sug_bs_time=34,qc_time=0,father_son_flag=0,pb_length=1896,logid=3542027227,seid=3542027843052803850,loc_cid=131,query_type=0,res_num=10,flow_sid=2,experiment_name=,experiment_sid=-1,print_ltr_log=0,recommend_insert_pos=-1,as_err=0,da_err=0,qc_err=0,sug_bs_err=0,bdrp_err=0'
   
    print line 
    m2 = re.match(".*(\[.*?\]) \"GET \/mapssu\?(.*?) .*SU \[(\{.*?\})\].*timeused (.*?),.*seid=(.*?),.*", line)  
    
    print m2.group(0)
    print m2.group(1)
    print m2.group(2) #解析出来uri
    print m2.group(3) #解析poilist 
    print m2.group(4)
    print m2.group(5)
    
    
    '''
    ？？从uri中怎么解析出来链接中key对应的value？
    
    '''
    
    
    
    
    
    
    
    '''
    {"q":"s","p":true,"t":0,
    "s":["松发肉骨茶(克拉码头店)$新加坡$20001$2276307f470b0e033210af07$"<703736361>,"新加坡<s_0>s<\/s_0>.e.a.海洋馆$新加坡$20001$dc5d08556a82e35a970319a4$"<180267759>]}
    
    ？？如何从上面的jison串中解析出来poilist？
    格式如下：
    
    松发肉骨茶(克拉码头店)$新加坡$20001$2276307f470b0e033210af07||703736361
    新加坡.e.a.海洋馆$新加坡$20001$dc5d08556a82e35a970319a4||180267759
    
    '''
    
    json_str = m2.group(3)
    
    parse_poi_from_string(json_str)
    
    line2 = "ltr_feature log  seid\x02123\x01i\x0212"
    line2 = line2[line2.find("seid"):]
    
    line_arr = line2.split("\x01")
    seid = line_arr[0].split("\x02")
    index = line_arr[1].split("\x02")
    print seid[0],seid[1],index[0],index[1]
    
    
    
    
    
    
    return;




def main():
    
    regular_use()
    
    return;



if __name__ == "__main__":
    
    main();
    
    
    