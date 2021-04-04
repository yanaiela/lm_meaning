import argparse
import operator
from glob import glob
from typing import List, Dict, Tuple
import random
from lm_meaning.explanation.explain import get_lm_preds, get_items, get_subj_obj_cooccurence_dic
from pararel.consistency.utils import read_json_file, read_jsonl_file

memorization_dir = 'data/output/spike_results/paraphrases/'
# paraphrases_dir = 'data/pattern_data/graphs_tense_json/'
paraphrases_dir = 'data/unpattern_data/'
cooccurrences_dir = 'data/output/spike_results/cooccurrences/'
# lm_dir = 'data/output/predictions_lm/bert_lama/'
lm_dir = 'data/output/predictions_lm/bert_lama_unpatterns/'


def cooccurrence(cooccurence_data: Dict, min_count: int, tuples_data: List[Tuple], lm_predictions) -> Dict:
    subj_obj_dic = get_subj_obj_cooccurence_dic(cooccurence_data)

    dic = {}
    for subj, obj in tuples_data:
        data_key = f'{subj}_SEP_{obj}'
        if data_key in cooccurence_data:
            obj_counts = subj_obj_dic[subj]
            biggest_obj, count = max(obj_counts.items(), key=operator.itemgetter(1))
            if biggest_obj == lm_predictions[subj][0] and count > min_count:
                dic[f'{subj}_{obj}'] = {'cooccurences': count}
            else:
                dic[f'{subj}_{obj}'] = {'cooccurences': -1}

        else:
            dic[f'{subj}_{obj}'] = {'cooccurences': None}
    return dic


def cooccurrence_counts(pattern_id, success):
    paraphrase_file = f'{paraphrases_dir}/{pattern_id}.jsonl'
    lm_file = f'{lm_dir}/{pattern_id}_bert-large-cased.json'
    cooccurrence_file = f'{cooccurrences_dir}/{pattern_id}.json'
    memorization_file = f'{memorization_dir}/{pattern_id}.json'

    paraphrases = read_jsonl_file(paraphrase_file)

    # for picking a random pattern
    relation_pattern = random.choice(paraphrases[1:])['pattern']
    # relation_pattern = paraphrases[0]['pattern']

    lm_results = read_json_file(lm_file)
    lm_predictions = get_lm_preds(lm_results[relation_pattern])

    cooccurrences = read_json_file(cooccurrence_file)
    memorization = read_json_file(memorization_file)

    pattern_data = get_items(memorization)

    cooccurrence_explained = cooccurrence(cooccurrences, 0, pattern_data,
                                                   lm_results[relation_pattern])

    explained = 0
    total = 0

    for k, tuple_explanation in cooccurrence_explained.items():
        # excluding cases where the LM does not get the answer right
        if success and (k not in lm_predictions or not lm_predictions[k]):
            continue
        if tuple_explanation['cooccurences']:
            if tuple_explanation['cooccurences'] != -1:
                explained += 1
            total += 1

    return explained, total


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
                explained_c, total_c = cooccurrence_counts(pattern, args.success)
                explained += explained_c
                total += total_c
            except:
                pass
    else:
        explained, total = cooccurrence_counts(args.pattern, args.success)
    print(explained / total)


if __name__ == '__main__':
    main()
