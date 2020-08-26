import argparse
import json
import os

import utils


def to_dic(data):
    dic = {}
    for row in data:
        dic[row['sub_label']] = row['obj_label']
    return dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--evidence1", type=str, help="")
    parse.add_argument("--evidence2", type=str, help="")
    parse.add_argument("--out", type=str, help="")
    args = parse.parse_args()

    # Load data
    if not os.path.exists(args.evidence1):
        raise ValueError('Relation "{}" does not exist in data.'.format(args.relation))
    data1 = utils.read_data(args.evidence1)
    data2 = utils.read_data(args.evidence2)

    dic1 = to_dic(data1)
    dic2 = to_dic(data2)

    filt_data = []
    for k, v in dic1.items():
        if dic2.get(k) == v:
            continue
        filt_data.append({"sub_label": k, "obj_label": v})

    with open(args.out, 'w') as f:
        for line in filt_data:
            json.dump(line, f)
            f.write('\n')


if __name__ == '__main__':
    main()
