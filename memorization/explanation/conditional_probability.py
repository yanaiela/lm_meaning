import argparse
import operator
from glob import glob
from typing import List, Dict, Tuple
import random
from lm_meaning.explanation.explain import get_lm_preds, get_items, get_subj_obj_cooccurence_dic
from pararel.consistency.utils import read_json_file, read_jsonl_file

memorization_dir = 'data/output/spike_results/paraphrases/'
paraphrases_dir = 'data/pattern_data/graphs_tense_json/'
cooccurrences_dir = 'data/output/spike_results/cooccurrences/'
lm_dir = 'data/output/predictions_lm/bert_lama/'


def cooccurrence(cooccurence_data: Dict, min_count: int, tuples_data: List[Tuple], lm_predictions):
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
                dic[f'{subj}_{obj}'] = {'cooccurences': None}

        else:
            dic[f'{subj}_{obj}'] = {'cooccurences': None}
    return dic


def cooccurrence_counts(pattern_id, all_patterns, success):
    paraphrase_file = f'{paraphrases_dir}/{pattern_id}.jsonl'
    lm_file = f'{lm_dir}/{pattern_id}_bert-large-cased.json'
    cooccurrence_file = f'{cooccurrences_dir}/{pattern_id}.json'
    memorization_file = f'{memorization_dir}/{pattern_id}.json'

    relation_pattern = [x['template'] for x in all_patterns if x['relation'] == pattern_id][0].replace(' .', '.')
    paraphrases = read_jsonl_file(paraphrase_file)
    # print(relation_pattern)
    # print([x['pattern'] for x in paraphrases])
    if relation_pattern not in [x['pattern'] for x in paraphrases]:
        relation_pattern = relation_pattern.replace('.', ' .')
    spike_pattern = [x['pattern'] for x in paraphrases if x['pattern'] == relation_pattern]
    if len(spike_pattern) == 0:
        print('can\'t fine relevant pattern')
        print(pattern_id)
        exit()

    # for picking a random pattern
    relation_pattern = random.choice(paraphrases)['pattern']

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
        # print(k, tuple_explanation)
        if tuple_explanation['cooccurences']:
            explained += 1
        total += 1

    return explained, total


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-p", "--pattern", type=str, help="pattern id",
                       default="P449")
    # parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
    #                    default="data/output/spike_results/preferences/P449.json")
    parse.add_argument("-s", "--success", action='store_true',
                       help="conditioning on the success of the LM on this pattern")

    args = parse.parse_args()

    relations_file = 'data/trex/data/relations.jsonl'

    all_relations = []
    for relation in glob('data/pattern_data/parsed/*.jsonl'):
        all_relations.append(relation.split('/')[-1].split('.')[0])

    all_patterns = read_jsonl_file(relations_file)

    if args.pattern == 'all':
        explained = 0
        total = 0
        for pattern in all_relations:
            if pattern in ['P166', 'P69', 'P54', 'P1923', 'P102', 'P31', 'P527', 'P1001']:
                continue
            explained_c, total_c = cooccurrence_counts(pattern, all_patterns, args.success)
            explained += explained_c
            total += total_c
    else:
        explained, total = cooccurrence_counts(args.pattern, all_patterns, args.success)
    print(explained / total)


if __name__ == '__main__':
    main()
