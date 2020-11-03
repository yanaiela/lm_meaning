import argparse
import json
from collections import defaultdict
from scipy.stats import entropy
from glob import glob


def get_obj2subj(relation_predictions):
    data = relation_predictions['data']
    preds = relation_predictions['predictions']

    predictions_dic = defaultdict(list)
    predictions, truth = [], []
    for data_row, pred_row in zip(data, preds):
        predictions_dic[pred_row[0]['token_str']].append(data_row['obj_label'])
        predictions.append(pred_row[0]['token_str'])
        truth.append(data_row['obj_label'])
    return predictions, truth


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-lm_preds", "--lm_preds", type=str, help="lm prediction file",
                       default="data/output/predictions_lm/lama/")

    args = parse.parse_args()

    for f_name in glob(args.lm_preds + '/*'):
        pattern_id = f_name.split('/')[-1].split('_')[0]
        with open(f_name, 'r') as f:
            lm_preds = json.load(f)

        orig_pattern = list(lm_preds.keys())[0]
        relation_predictions = lm_preds[orig_pattern]

        predictions, truths = get_obj2subj(relation_predictions)

        print(pattern_id, "& {:.2f} & {:.2f} & {} & {} \\\\".format(
            entropy([predictions.count(x) / len(predictions) for x in set(predictions)],
                    base=2),
            entropy([truths.count(x) / len(truths) for x in set(truths)],
                    base=2),
            len(set(predictions)),
            len(set(truths))))


if __name__ == '__main__':
    main()
