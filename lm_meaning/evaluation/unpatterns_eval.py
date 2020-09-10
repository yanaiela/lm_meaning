import argparse

from scipy.stats import wilcoxon
import wandb

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


def analyze_lm_unpattern(lm_results, base_pattern, alternative_patterns):

    cooccurrence = 0

    base_pattern_acc = 0
    other_relations_acc = 0

    main_relation_success = []
    other_relation_success = []

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
                else:
                    other_relation_success.append(0)
            if base_pattern in lm_results[key]:
                base_pattern_acc += 1

    print('cooccurrences (at least one of the patterns captured the object): {}/{}'.format(cooccurrence,
          len(lm_results)))
    print('base pattern success: {}'.format(base_pattern_acc / cooccurrence))
    print('other patterns "success": {}'.format((other_relations_acc / len(alternative_patterns)) / cooccurrence))

    wandb.run.summary['cooccurrence'] = cooccurrence
    wandb.run.summary['total_tuples'] = len(lm_results)
    wandb.run.summary['base_acc'] = base_pattern_acc / cooccurrence
    wandb.run.summary['false_acc'] = (other_relations_acc / len(alternative_patterns)) / cooccurrence
    wandb.run.summary['pval'] = wilcoxon(main_relation_success, other_relation_success, alternative='greater').pvalue

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

    lm_patterns = [x for x in read_jsonline_file(args.lm_patterns)]
    # print(lm_patterns)
    base_pattern = [x['pattern'] for x in lm_patterns if x['base'] is True][0]
    alternative_patterns = [x['pattern'] for x in lm_patterns if x['base'] is False]

    lm_raw_results = read_json_file(args.lm_file)

    lm_results = parse_lm_results(lm_raw_results)

    analyze_lm_unpattern(lm_results, base_pattern, alternative_patterns)


if __name__ == '__main__':
    main()
