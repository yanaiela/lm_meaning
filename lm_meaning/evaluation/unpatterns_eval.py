import argparse

from collections import defaultdict
import operator
from scipy.stats import wilcoxon
import wandb
from typing import Dict, List
from lm_meaning.evaluation.paraphrase_comparison import read_json_file, read_jsonline_file
from lm_meaning.evaluation.spike_lm_eval import parse_lm_results


def log_wandb(args):
    pattern = args.lm_patterns.split('/')[-1].split('.')[0]
    lm = args.lm_file.split('/')[-1].split('.')[0].split('_')[-1]

    config = dict(
        pattern=pattern,
        lm=lm
    )

    wandb.init(
        name=f'{pattern}_unpattern_eval_{lm}',
        project="memorization",
        tags=["eval", pattern, 'unpattern', lm],
        config=config,
    )


def analyze_lm_unpattern(lm_results: Dict, base_pattern: str, alternative_patterns: List[str], pattern_id: str):

    cooccurrence = 0

    base_pattern_acc = 0
    other_relations_acc = 0

    main_relation_success = []
    other_relation_success = []

    alternative_patterns_success = defaultdict(int)

    # Going over all the subj-obj pairs
    for key, vals in lm_results.items():

        # This filters out tuples that none of the relations captured them
        # The reason to filter based on that is to filter cases where the signal is so weak,
        # that no information was captured.
        if len(vals) > 0:
            cooccurrence += 1

            # for spike_pattern in vals:
            for lm_pattern in alternative_patterns:

                # counting if the "base pattern" is successfully predicted by the LM
                if base_pattern in lm_results[key]:
                    main_relation_success.append(1)
                else:
                    main_relation_success.append(0)

                # counting
                if lm_pattern in lm_results[key]:
                    other_relation_success.append(1)
                    other_relations_acc += 1
                    alternative_patterns_success[lm_pattern] += 1
                else:
                    other_relation_success.append(0)
            if base_pattern in lm_results[key]:
                base_pattern_acc += 1

    most_confusing_items = {}
    for key, vals in lm_results.items():
        other_patterns_success = len(vals)
        # not counting the "true" pattern
        if base_pattern in vals:
            other_patterns_success -= 1
        most_confusing_items[key] = other_patterns_success

    print('cooccurrences (at least one of the patterns captured the object): {}/{}'.format(cooccurrence,
          len(lm_results)))
    print('base pattern success: {}'.format(base_pattern_acc / cooccurrence))
    print('other patterns "success": {}'.format((other_relations_acc / len(alternative_patterns)) / cooccurrence))

    wandb.run.summary['cooccurrence'] = cooccurrence
    wandb.run.summary['num_patterns'] = len(alternative_patterns)
    wandb.run.summary['total_tuples'] = len(lm_results)
    wandb.run.summary['base_acc'] = base_pattern_acc / cooccurrence
    wandb.run.summary['false_acc'] = (other_relations_acc / len(alternative_patterns)) / cooccurrence
    if sum(main_relation_success) == 0 or sum(other_relation_success) == 0:
        wandb.run.summary['pval'] = -1
        return
    wandb.run.summary['pval'] = wilcoxon(main_relation_success, other_relation_success, alternative='greater').pvalue

    best_pattern = max(alternative_patterns_success.items(), key=operator.itemgetter(1))[0]
    wandb.run.summary['best_pattern'] = best_pattern
    wandb.run.summary['base_pattern'] = base_pattern
    wandb.run.summary['best_pattern_acc'] = alternative_patterns_success[best_pattern] / cooccurrence

    # logging a table of the alternated patterns and their performance
    table = wandb.Table(columns=["Pattern", "Acc"])
    for k, v in alternative_patterns_success.items():
        table.add_data(k, v / cooccurrence)
    wandb.log({"patterns_results": table})

    table = wandb.Table(columns=["Tuple", "Patterns"])
    counter = 0
    for w in sorted(most_confusing_items, key=most_confusing_items.get, reverse=True):
        table.add_data(w, most_confusing_items[w])
        counter += 1
        if counter >= 50:
            break
    wandb.log({"confusing_tuples": table})

    # print('lm acc: {}'.format(sum(main_relation_sucess) / len(main_relation_sucess)))
    # print('spike acc: {}'.format(sum(other_relation_success) / len(other_relation_success)))
    print(wilcoxon(main_relation_success, other_relation_success, alternative='greater').pvalue)
    print(main_relation_success[:30])
    print(other_relation_success[:30])


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file",
                       default="data/predictions_lm/P449_bert-large-cased.json")
    parse.add_argument("-lm_patterns", "--lm_patterns", type=str, help="spike pattern",
                       default="data/spike_patterns/P449.txt")

    args = parse.parse_args()
    log_wandb(args)

    pattern = args.lm_patterns.split('/')[-1].split('.')[0]

    lm_patterns = [x for x in read_jsonline_file(args.lm_patterns)]
    # print(lm_patterns)
    base_pattern = [x['pattern'] for x in lm_patterns if x['base'] is True][0]
    alternative_patterns = [x['pattern'] for x in lm_patterns if x['base'] is False]

    lm_raw_results = read_json_file(args.lm_file)

    lm_results = parse_lm_results(lm_raw_results)

    analyze_lm_unpattern(lm_results, base_pattern, alternative_patterns, pattern)


if __name__ == '__main__':
    main()
