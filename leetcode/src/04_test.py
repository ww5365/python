# coding=utf-8
import sys
for line in sys.stdin:
    line = line.strip()
    tmp = set()
    li_line = list(line)
    valid_len = 0
    valid_idx = -1
    for i in range(len(li_line)):
        if li_line[i] not in tmp:
            valid_idx += 1
            li_line[valid_idx] = li_line[i]
            tmp.add(li_line[i])
            # print(tmp)
    print("valid len:", valid_idx)
    print("str:", li_line)
    print(li_line[0:valid_idx+1])

    print("".join(li_line[0:valid_idx + 1]))
