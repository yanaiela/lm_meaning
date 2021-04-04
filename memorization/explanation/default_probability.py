import argparse
import random
from glob import glob
from typing import Tuple, Dict, List

from pararel.consistency.utils import read_json_file, read_jsonl_file

default_dir = 'data/output/spike_results/preferences/'
paraphrases_dir = 'data/pattern_data/parsed/'
cooccurrences_dir = 'data/output/spike_results/cooccurrences/'
lm_dir = 'data/output/predictions_lm/bert_lama/'
trex_dir = 'data/trex/data/TREx/'


def default_object(default_objects: Dict, pattern: str, trex_objects: List[str], trex: Dict) -> Dict[str, str]:
    vals = default_objects[pattern]
    top_preference = list(dict(sorted(vals.items(), key=lambda item: item[1], reverse=True)))
    top = None
    for x in top_preference:
        if x in trex_objects:
            top = x
            break

    dic = {}
    for row in trex:
        subj = row['sub_label']
        obj = row['obj_label']
        dic[f'{subj}_{obj}'] = top

    return dic


def default_counts(pattern_id: str, success: bool) -> Tuple[int, int, int]:
    paraphrase_file = f'{paraphrases_dir}/{pattern_id}.jsonl'
    lm_file = f'{lm_dir}/{pattern_id}_bert-large-cased.json'
    default_file = f'{default_dir}/{pattern_id}.json'

    trex_file = f'{trex_dir}/{pattern_id}.jsonl'
    trex_data = read_jsonl_file(trex_file)
    objects = list(set([x['obj_label'] for x in trex_data]))

    paraphrases = read_jsonl_file(paraphrase_file)

    # for picking a random pattern
    # pattern = paraphrases[0]
    pattern = random.choice(paraphrases)
    # spike_pattern = pattern['spike_query']
    relation_pattern = pattern['pattern']
    # relation_pattern = paraphrases[2]['pattern']

    lm_results = read_json_file(lm_file)
    # lm_predictions = get_lm_preds(lm_results[relation_pattern])

    default_objects = read_json_file(default_file)

    default_explained = default_object(default_objects, relation_pattern, objects, trex_data)

    default_occurence = 0
    explained = 0
    total = 0

    for k, tuple_explanation in default_explained.items():
        # excluding cases where the LM does not get the answer right
        # if success and (k not in lm_predictions or not lm_predictions[k]):
        #     continue
        if tuple_explanation:
            if k.split('_')[1] == tuple_explanation:
            # if k in lm_predictions:
                default_occurence += 1
            if lm_results[relation_pattern][k.split('_')[0]][0] == tuple_explanation:
                explained += 1
            total += 1

    return default_occurence, explained, total


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
        default_occ = 0
        explained = 0
        total = 0
        for pattern in all_relations:
            if pattern in ['P166', 'P69', 'P54', 'P1923', 'P102', 'P31', 'P527', 'P1001']:
                continue
            try:
                default_c, explained_c, total_c = default_counts(pattern, args.success)
                default_occ += default_c
                explained += explained_c
                total += total_c
            except:
                pass
    else:
        default_occ, explained, total = default_counts(args.pattern, args.success)
    print('majority for most common value', default_occ / total)
    print('prediction for most common value', explained / total)
    print(explained, total)


if __name__ == '__main__':
    main()
