import argparse
import random
from glob import glob
from typing import Tuple

from pararel.consistency.utils import read_json_file, read_jsonl_file

paraphrases_dir = 'data/pattern_data/graphs_tense_json/'
pred_pattern_dir = 'data/output/predictions_lm/bert_lama/'


def ignore_relation(pattern_id: str) -> Tuple[int, int]:
    pattern_pred_file = f'{pred_pattern_dir}/{pattern_id}_bert-large-cased.json'

    paraphrase_file = f'{paraphrases_dir}/{pattern_id}.jsonl'
    paraphrases = read_jsonl_file(paraphrase_file)
    base_pattern = paraphrases[0]['pattern']

    paraphrase_pattern = random.choice(paraphrases[1:])['pattern']

    pattern_results = read_json_file(pattern_pred_file)

    same = 0
    total = 0
    for subj, (pred, obj) in pattern_results[base_pattern].items():
        if pattern_results[paraphrase_pattern][subj][0] == pred:
            same += 1
        total += 1
    return same, total


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-p", "--pattern", type=str, help="pattern id",
                       default="P449")
    parse.add_argument("-s", "--success", action='store_true',
                       help="conditioning on the success of the LM on this pattern")

    args = parse.parse_args()

    all_relations = []
    for relation in glob('data/pattern_data/parsed/*.jsonl'):
        all_relations.append(relation.split('/')[-1].split('.')[0])

    if args.pattern == 'all':
        explained = 0
        total = 0
        for pattern in all_relations:
            if pattern in ['P166', 'P69', 'P54', 'P1923', 'P102', 'P31', 'P527', 'P1001']:
                continue
            try:
                explained_c, total_c = ignore_relation(pattern)
                explained += explained_c
                total += total_c
            except:
                pass
    else:
        explained, total = ignore_relation(args.pattern)
    print(explained / total)


if __name__ == '__main__':
    main()
