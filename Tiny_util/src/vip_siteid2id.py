# -*- coding:utf-8 -*-

import os
import sys
import codecs
import json
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - [line:%(lineno)s] - %(levelname)s: %(message)s")

if __name__ == '__main__':

    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cur_path, 'data')
    # print(data_path)
    grade_score_file = os.path.join(data_path, "grade_score.json")
    logging.debug("begin to process [%s]" % (grade_score_file))

    vip_siteid_set = set()
    vip_dict = dict()
    with codecs.open(grade_score_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            # print(line)
            js = json.loads(line)
            site_id = js.get("site_id", "")
            quality_score = js.get("quality_score", 0)

            if site_id not in vip_dict:
                vip_dict[site_id] = quality_score

            if site_id not in vip_siteid_set:
                vip_siteid_set.add(site_id)

    logging.debug("process [%s] is over" % (grade_score_file))

    print("vip_dict size:[%d] vip_siteid_set size:[%d]" %
          (len(vip_dict), len(vip_siteid_set)))

    dict_file = os.path.join(data_path, "id2siteid.test")
    siteid_set = set()
    siteid2id_dict = dict()
    with codecs.open(dict_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            id = line[0]
            site_id = line[1]
            if site_id in vip_siteid_set:
                siteid_set.add(site_id)
                if site_id not in siteid2id_dict:
                    siteid2id_dict[site_id] = id

    not_exsit_siteid = vip_siteid_set.difference(siteid_set)

    print("dict file size: ", len(siteid_set), len(siteid2id_dict))
    print("not exsit siteid size: ", len(not_exsit_siteid))

    result_file_path = os.path.join(data_path, "siteid2id_res.txt")
    result_file = codecs.open(result_file_path, 'w', encoding='utf-8')
    for key, value in siteid2id_dict.items():
        result_file.write(key + "\t" + value + "\n")
    result_file.close()

    not_exsit_file_path = os.path.join(data_path, "not_exsit_siteid.txt")
    not_exsit_file = codecs.open(not_exsit_file_path, 'w', encoding='utf-8')
    for key in not_exsit_siteid:
        not_exsit_file.write(key + "\n")
    not_exsit_file.close()
