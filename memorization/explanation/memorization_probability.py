import argparse
from glob import glob
from typing import Tuple

from lm_meaning.explanation.explain import get_lm_preds, explain_memorization
from pararel.consistency.utils import read_json_file, read_jsonl_file

memorization_dir = 'data/output/spike_results/paraphrases/'
paraphrases_dir = 'data/pattern_data/parsed/'
cooccurrences_dir = 'data/output/spike_results/cooccurrences/'
lm_dir = 'data/output/predictions_lm/bert_lama/'


def memorization_counts(pattern_id: str, effect: bool) -> Tuple[int, int]:
    paraphrase_file = f'{paraphrases_dir}/{pattern_id}.jsonl'
    lm_file = f'{lm_dir}/{pattern_id}_bert-large-cased.json'
    memorization_file = f'{memorization_dir}/{pattern_id}.json'

    paraphrases = read_jsonl_file(paraphrase_file)

    # for picking a random pattern
    pattern = paraphrases[0]
    # pattern = random.choice(paraphrases)
    spike_pattern = pattern['spike_query']
    relation_pattern = pattern['pattern']
    # relation_pattern = paraphrases[2]['pattern']

    lm_results = read_json_file(lm_file)
    lm_predictions = get_lm_preds(lm_results[relation_pattern])

    memorization = read_json_file(memorization_file)

    memorization_explained = explain_memorization(memorization, spike_pattern)

    explained = 0
    total = 0

    for k, tuple_explanation in memorization_explained.items():
        # when effect is True, measuring the effect when the pattern appeared in the training data
        # otherwise, measuring it when it's not in the training data
        if (tuple_explanation['memorization'] is not None) == effect:
            if k in lm_predictions and lm_predictions[k]:
                explained += 1
            total += 1

    return explained, total


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-p", "--pattern", type=str, help="pattern id",
                       default="P449")
    parse.add_argument("-e", "--effect", action='store_true',
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
                explained_c, total_c = memorization_counts(pattern, args.effect)
                explained += explained_c
                total += total_c
            except:
                pass
    else:
        explained, total = memorization_counts(args.pattern, args.effect)
    print(explained / total)
    print(explained, total)


if __name__ == '__main__':
    main()
