import argparse
import glob
import json
from collections import defaultdict

from lm_meaning.spike.utils import get_relations_data


def get_data(relation_file, cooccurrence_file):
    relations = get_relations_data(relation_file)
    subj_dic = {}
    for row in relations:
        # Multiple subjects exists - keeping just the last one
        # assert row['sub_label'] not in subj_dic, (subj_dic[row['sub_label']], (row['sub_label'], row['obj_label']))
        subj_dic[row['sub_label']] = row['obj_label']

    with open(cooccurrence_file, 'r') as f:
        cooccurrences = json.load(f)

    cooccurrences_dic = defaultdict(dict)
    for key, count in cooccurrences.items():
        subj, obj = key.split('_SEP_')
        cooccurrences_dic[subj][obj] = count

    return subj_dic, cooccurrences_dic


def eval_cooccurrences(subj_dic, cooccurrences_dic):
    ranks = []
    tops = 0

    for subj, obj in subj_dic.items():
        obj_counts = cooccurrences_dic[subj]
        for ind, cur_obj in enumerate(sorted(obj_counts, key=obj_counts.get, reverse=True)):
            if cur_obj == obj:
                ranks.append(ind + 1)
                if ind == 0:
                    tops += 1

    return tops / len(subj_dic), sum(ranks) / len(ranks)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-data_path", "--data_path", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/trex/data/TREx/")
    parse.add_argument("-cooccurrence_path", "--cooccurrence_path", type=str, help="cooccurences between subjects"
                                                                                   "and objects file",
                       default="data/output/spike_results/cooccurrences/")

    args = parse.parse_args()

    for file_path in glob.glob(args.cooccurrence_path + '/*.json'):
        relation = file_path.split('/')[-1].split('.')[0]

        subj_dic, cooccurrences_dic = get_data(args.data_path + '/' + relation + '.jsonl',
                                               args.cooccurrence_path + '/' + relation + '.json')

        tops, ranks = eval_cooccurrences(subj_dic, cooccurrences_dic)
        # printing the results in latex format
        print(relation, '&', "{:.2f}".format(tops), '&', "{:.2f}".format(ranks), '\\\\')


if __name__ == '__main__':
    main()
