# -*- coding:utf-8 -*-

import os
import sys
import codecs
import json


if __name__ == '__main__':

    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cur_path, 'data')
    print(data_path)

    dict_file = os.path.join(data_path, "importance1.txt")
    site_id_set = set()
    with codecs.open(dict_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            site_id_set.add(line)

    print("dict file size: ", len(site_id_set))

    grade_score_file = os.path.join(data_path, "grade_score.json")
    result_file_path = os.path.join(data_path, "dict.txt")
    result_file = codecs.open(result_file_path, 'w', encoding='utf-8')

    with codecs.open(grade_score_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            # print(line)
            js = json.loads(line)
            site_id = js.get("site_id", "")
            quality_score = js.get("quality_score", 0)
            if site_id in site_id_set and quality_score >= 3.0:
                print(site_id, quality_score)
                result_file.write(site_id + " " + str(3) + "\n")

    print("result file finished!")
    result_file.close()
