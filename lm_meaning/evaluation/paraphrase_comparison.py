import argparse
import logging
import json
import jsonlines
from collections import defaultdict
from typing import Dict
from scipy.stats import wilcoxon

# statistical reasoning choice and used code comes from the following:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
# https://www.aclweb.org/anthology/P18-1128.pdf
# http://nlp.cs.berkeley.edu/pubs/BergKirkpatrick-Burkett-Klein_2012_Significance_paper.pdf


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_jsonline_file(in_f):
    with jsonlines.open(in_f, 'r') as f:
        lines = list(f)
    return lines


def read_json_file(in_f) -> Dict:
    with open(in_f, 'r') as f:
        data = f.read()
    return json.loads(data)


def evaluate(lm, origin_pattern):
    outputs = {}
    results_dic = defaultdict(dict)

    for pattern, data in lm.items():
        print(pattern)
        gold = data['data']
        preds = data['predictions']

        pattern_success = 0
        for gold_dic, top_k_pred in zip(gold, preds):
            key = '_TUPLE_'.join([gold_dic['sub_label'], gold_dic['obj_label']])
            if gold_dic['obj_label'] == top_k_pred[0]:
                pattern_success += 1
                results_dic[pattern][key] = 1
            else:
                results_dic[pattern][key] = 0
        outputs[pattern] = [pattern_success, len(preds), pattern_success / len(preds)]
    print(outputs)

    patterns = results_dic.keys()
    anchor_results = []
    pattern_results = defaultdict(list)

    test_patterns = set(patterns)
    test_patterns.discard(origin_pattern)
    for tuple, origin_value in results_dic[origin_pattern].items():
        if origin_value != 1: continue
        for pattern in test_patterns:
            pattern_results[pattern].append(results_dic[pattern][tuple])
        anchor_results.append(origin_value)

    for pattern, test_values in pattern_results.items():
        print('pattern: {}, wilcoxon test: {}'.format(pattern,
                                                      wilcoxon(anchor_results, test_values,
                                                               alternative='greater').pvalue))
        print(sum(test_values), len(test_values), sum(test_values) / len(test_values))


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-pred_file", "--pred_file", type=str, help="lm prediction file",
                       default="predictions_TREx/results_bert-base-cased.json")
    parse.add_argument("-gold_file", "--gold_file", type=str, help="gold relations file",
                       default="data/text_evidence/P449.jsonl")
    parse.add_argument("-pattern", "--pattern", type=str, help="relation pattern",
                       default="[X] was originally aired on [Y] .")

    args = parse.parse_args()

    rules_ans = read_jsonline_file(args.gold_file)
    lm_ans = read_json_file(args.pred_file)

    pattern = args.pattern
    print(pattern)

    evaluate(lm_ans, pattern)


if __name__ == '__main__':
    main()
