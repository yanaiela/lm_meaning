import argparse
import json
from collections import defaultdict


def get_obj2subj(relation_predictions):
    data = relation_predictions['data']
    preds = relation_predictions['predictions']

    predictions_dic = defaultdict(list)
    for data_row, pred_row in zip(data, preds):
        predictions_dic[pred_row[0]['token_str']].append(data_row['obj_label'])
    return predictions_dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-lm_preds", "--lm_preds", type=str, help="lm prediction file",
                       default="data/output/predictions_lm/lama/")

    args = parse.parse_args()

    with open(args.lm_preds, 'r') as f:
        lm_preds = json.load(f)

    orig_pattern = list(lm_preds.keys())[0]
    relation_predictions = lm_preds[orig_pattern]

    obj2subjs = get_obj2subj(relation_predictions)

    max_num = -1
    largest_pred_list = []
    most_predicted_obj = None
    for pred_obj, vals in obj2subjs.items():
        print(pred_obj, vals.count(pred_obj), len(vals))
        if len(vals) > max_num:
            largest_pred_list = vals
            most_predicted_obj = pred_obj
            max_num = len(largest_pred_list)

    # print(max_num)
    # print(largest_pred_list.count(most_predicted_obj))


if __name__ == '__main__':
    main()
