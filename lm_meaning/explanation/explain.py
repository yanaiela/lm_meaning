import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

from lm_meaning.utils import read_json_file, read_jsonl_file


def get_items(memorization_data: Dict):
    data = []
    for obj, subj_dic in memorization_data.items():
        for subj, _ in subj_dic.items():
            data.append((subj, obj))
    return data


def get_lm_preds(lm_preds):
    pred_dic = {}
    for data, preds in zip(lm_preds['data'], lm_preds['predictions']):
        subj = data['sub_label']
        obj = data['obj_label']
        key = '_'.join([subj, obj])
        if preds[0]['token_str'] == obj:
            pred_dic[key] = True
        else:
            pred_dic[key] = False
    return pred_dic


def explain_memorization(memorization_data: Dict, pattern: str):
    dic = {}
    for obj, subj_dic in memorization_data.items():
        for subj, explanation in subj_dic.items():
            dic[f'{subj}_{obj}'] = {'memorization': None}
            for spike_pattern, sentence in explanation.items():
                if spike_pattern == pattern:
                    dic[f'{subj}_{obj}'] = {'memorization': ' '.join(sentence)}
            # dic[f'{subj}_{obj}'] = {'memorization': explanation if len(explanation) != 0 else None}
    return dic


def explain_cooccurrences(cooccurence_data: Dict, min_count: int, tuples_data: List[Tuple]):
    dic = {}
    for subj, obj in tuples_data:
        data_key = f'{subj}_SEP_{obj}'
        if data_key in cooccurence_data:
            count = cooccurence_data[data_key]
            if count > min_count:
                dic[f'{subj}_{obj}'] = {'cooccurences': count}
            else:
                dic[f'{subj}_{obj}'] = {'cooccurences': None}
        else:
            dic[f'{subj}_{obj}'] = {'cooccurences': None}
    return dic


def explain_preference_bias(preference_bias_data: List, top_k: int, tuples_data: List[Tuple]):
    dic = {}
    for subj, obj in tuples_data:
        if obj in preference_bias_data[:top_k]:
            dic[f'{subj}_{obj}'] = {'preference': preference_bias_data.index(obj)}
        else:
            dic[f'{subj}_{obj}'] = {'preference': None}
    return dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-bias_file", "--bias_file", type=str, help="preference bias file")
    parse.add_argument("-memorization_file", "--memorization_file", type=str, help="")
    parse.add_argument("-paraphrase_file", "--paraphrase_file", type=str, help="")
    parse.add_argument("-cooccurrence_file", "--cooccurrence_file", type=str, help="")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file")
    parse.add_argument("-relation_file", "--relation_file", type=str, help="lama's relations file")

    args = parse.parse_args()

    pattern_id = args.memorization_file.split('/')[-1].split('.')[0]

    all_patterns = read_jsonl_file(args.relation_file)
    relation_pattern = [x['template'] for x in all_patterns if x['relation'] == pattern_id][0].replace(' .', '.')
    paraphrases = read_jsonl_file(args.paraphrase_file)
    if pattern_id == 'P449':
        spike_pattern = "<>subject:Lost $was $aired $on object:[w={}]ABC."
    else:
        spike_pattern = [x['spike_query'] for x in paraphrases if x['pattern'] == relation_pattern][0]
    print('here')
    lm_results = read_json_file(args.lm_file)
    lm_predictions = get_lm_preds(list(lm_results.values())[0])

    preference_bias = read_json_file(args.bias_file)[pattern_id]
    cooccurrences = read_json_file(args.cooccurrence_file)
    memorization = read_json_file(args.memorization_file)

    pattern_data = get_items(memorization)

    memorization_explained = explain_memorization(memorization, spike_pattern)
    cooccurrence_explained = explain_cooccurrences(cooccurrences, 100, pattern_data)
    preference_bias_explained = explain_preference_bias(preference_bias, 5, pattern_data)

    explanations = {}
    for k, v in memorization_explained.items():
        explanations[k] = {**v, **cooccurrence_explained[k], **preference_bias_explained[k]}

    # print(len(explanations))

    n_explanations = 0
    explanation_type = defaultdict(int)
    lm_correct_count = 0
    for k, tuple_explanation in explanations.items():
        # excluding cases where the LM does not get the answer right
        if not lm_predictions[k]:
            continue
        lm_correct_count += 1
        found_explanation = False
        for specific_explanation, val in tuple_explanation.items():
            if val:
                found_explanation = True
                explanation_type[specific_explanation] += 1
        if found_explanation:
            n_explanations += 1
    print(n_explanations, lm_correct_count)
    print(explanation_type)


if __name__ == '__main__':
    main()
