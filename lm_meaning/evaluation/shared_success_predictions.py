"""
Filter data to tuples that all models predicted correctly
"""
import argparse

from pararel.consistency.entailment_probe import get_first_object
from lm_meaning.evaluation.paraphrase_comparison import read_json_file
from pararel.consistency.utils import read_jsonl_file


# def log_wandb(args):
#     pattern = args.trex.split('/')[-1].split('.')[0]
#     lm = args.lm_file.split('/')[-1].split('.')[0].split('_')[-1]
#     config = dict(
#         pattern=pattern,
#         lm=lm
#     )
#
#     wandb.init(
#         name=f'{pattern}_entailment_probe_{lm}',
#         project="memorization",
#         tags=["eval", pattern, 'paraphrase', lm],
#         config=config,
#     )


def find_joint_data(results_per_lm, possible_objects):
    output_dic = {}
    for lm_results in results_per_lm:
        for pattern, dic in lm_results.items():
            for data, preds in zip(dic['data'], dic['predictions']):
                subj = data['sub_label']
                obj = data['obj_label']
                key = '_SPLIT_'.join([subj, obj])
                if key not in output_dic:
                    output_dic[key] = []
                first_object = get_first_object(preds, possible_objects)
                if first_object == obj:
                    output_dic[key].append(pattern)
    return output_dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-trex", "--trex", type=str, help="trex data file",
                       default="data/trex/data/TREx/P449.jsonl")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file",
                       default="data/predictions_lm/P449_{}.json")
    encoders = ['bert-base-cased',
                'bert-large-cased',
                'bert-large-cased-whole-word-masking',
                'roberta-base',
                'roberta-large',
                'albert-base-v2',
                'albert-xxlarge-v2'
                ]

    args = parse.parse_args()
    # log_wandb(args)

    encoders_data = []
    for enc in encoders:
        encoders_data.append(read_json_file(args.lm_file.format(enc)))


    data = read_jsonl_file(args.trex)
    subj_obj = {}
    for row in data:
        subj_obj[row['sub_label']] = row['obj_label']

    all_objects = list(set(subj_obj.values()))

    lm_results = find_joint_data(encoders_data, all_objects)

    joint = 0
    overall = 0
    for k, patterns in lm_results.items():
        for unique_pattern in set(patterns):
            c = patterns.count(unique_pattern)
            if c == len(encoders):
                joint += 1
            overall += 1

    print('join: {}, overall: {}'.format(joint, overall))


if __name__ == '__main__':
    main()
