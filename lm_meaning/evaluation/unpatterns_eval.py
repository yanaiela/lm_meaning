import argparse

from scipy.stats import wilcoxon

from lm_meaning.evaluation.paraphrase_comparison import read_json_file, read_jsonline_file
from lm_meaning.evaluation.spike_lm_eval import read_txt_lines, parse_lm_results, match_spike_lm_patterns


def analyze_lm_unpattern(lm_results, base_pattern, alternative_patterns):

    lm_acc = 0
    spike_acc = 0

    main_relation_sucess = []
    other_relation_success = []

    # Going over all the subj-obj pairs
    for key, vals in lm_results.items():
        if len(vals) > 0:
            spike_acc += 1

            # for spike_pattern in vals:
            for lm_pattern in alternative_patterns:

                # counting if the "base pattern" is successfully predicted by the LM
                if base_pattern in lm_results[key]:
                    main_relation_sucess.append(1)
                else:
                    main_relation_sucess.append(0)

                # counting
                if lm_pattern in lm_results[key]:
                    other_relation_success.append(1)
                else:
                    other_relation_success.append(0)

            # if len(lm_results[key]) > 0:
            #     print(key, 'lm:', lm_results[key])
        # if len(lm_results[key]) > 0:
        #     lm_acc += 1

    print('lm acc: {}'.format(sum(main_relation_sucess) / len(main_relation_sucess)))
    print('spike acc: {}'.format(sum(other_relation_success) / len(other_relation_success)))
    print(wilcoxon(main_relation_sucess, other_relation_success, alternative='two-sided').pvalue)
    print(main_relation_sucess[:30])
    print(other_relation_success[:30])


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file",
                       default="data/predictions_lm/P449_bert-large-cased.json")
    parse.add_argument("-lm_patterns", "--lm_patterns", type=str, help="spike pattern",
                       default="data/spike_patterns/P449.txt")

    args = parse.parse_args()

    lm_patterns = [x for x in read_jsonline_file(args.lm_patterns)]
    # print(lm_patterns)
    base_pattern = [x['pattern'] for x in lm_patterns if x['base'] is True][0]
    alternative_patterns = [x['pattern'] for x in lm_patterns if x['base'] is False]

    lm_raw_results = read_json_file(args.lm_file)

    lm_results = parse_lm_results(lm_raw_results)

    analyze_lm_unpattern(lm_results, base_pattern, alternative_patterns)


if __name__ == '__main__':
    main()
