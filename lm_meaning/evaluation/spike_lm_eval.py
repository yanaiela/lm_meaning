import argparse

from scipy.stats import wilcoxon

from lm_meaning.evaluation.paraphrase_comparison import read_json_file, read_jsonline_file


def read_txt_lines(in_f):
    with open(in_f, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def match_spike_lm_patterns(spike_patterns, lm_patterns):
    spike2lm = {}
    for spike, lm in zip(spike_patterns, lm_patterns):
        spike2lm[spike] = lm
    return spike2lm


def parse_spike_results(spike_results):
    output_dic = {}
    for obj, dic in spike_results.items():
        for subj, inner_dic in dic.items():
            output_dic['_SPLIT_'.join([subj, obj])] = list(inner_dic.keys())
    return output_dic


def parse_lm_results(lm_results):
    output_dic = {}
    for pattern, dic in lm_results.items():
        for data, preds in zip(dic['data'], dic['predictions']):
            subj = data['sub_label']
            obj = data['obj_label']
            key = '_SPLIT_'.join([subj, obj])
            if key not in output_dic:
                output_dic[key] = []
            if preds[0]['token_str'] == obj:
                output_dic[key].append(pattern)
    return output_dic


def analyze_results(lm_results, spike_results, spike2lm):

    lm2spike = {v: k for k, v in spike2lm.items()}

    lm_acc = 0
    spike_acc = 0

    spike_sucess = []
    lm_success = []

    # Going over all the subj-obj pairs
    for key, vals in spike_results.items():
        if len(vals) > 0:
            spike_acc += 1

            for spike_pattern in vals:
                # spike_sucess.append(1)
                for lm_pattern in spike2lm.values():
                    # in case the lm pattern was also found in wikipedia (with the spike patter), ignore
                    if lm2spike[lm_pattern] in vals and spike2lm[spike_pattern] != lm_pattern:
                        continue

                    # counting if the "base pattern" is successfully predicted by the LM
                    if spike2lm[spike_pattern] in lm_results[key]:
                        spike_sucess.append(1)
                    else:
                        spike_sucess.append(0)

                    # counting
                    if lm_pattern in lm_results[key]:
                        lm_success.append(1)
                    else:
                        lm_success.append(0)

            if len(lm_results[key]) > 0:
                print(key, 'spike:', [spike2lm[x] for x in vals], 'lm:', lm_results[key])
        if len(lm_results[key]) > 0:
            lm_acc += 1

    print('lm acc: {}'.format(lm_acc / len(lm_results)))
    print('spike acc: {}'.format(spike_acc / len(spike_results)))
    print(wilcoxon(spike_sucess, lm_success, alternative='greater').pvalue)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file",
                       default="data/predictions_lm/P449_bert-large-cased.json")
    parse.add_argument("-spike_file", "--spike_file", type=str, help="spike results file",
                       default="data/spike_results/P449.json")
    parse.add_argument("-lm_patterns", "--lm_patterns", type=str, help="lm patterns",
                       default="data/lm_relations/P449.jsonl")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="spike pattern",
                       default="data/spike_patterns/P449.txt")

    args = parse.parse_args()

    lm_patterns = [x['pattern'] for x in read_jsonline_file(args.spike_patterns)]
    # lm_patterns = read_jsonline_file(args.lm_patterns)
    # spike_patterns = read_txt_lines(args.spike_patterns)
    spike_patterns = [x['spike_query'] for x in read_jsonline_file(args.spike_patterns)]
    spike2lm = match_spike_lm_patterns(spike_patterns, lm_patterns)

    lm_raw_results = read_json_file(args.lm_file)
    spike_raw_results = read_json_file(args.spike_file)

    lm_results = parse_lm_results(lm_raw_results)
    spike_results = parse_spike_results(spike_raw_results)

    # print('hi')

    analyze_results(lm_results, spike_results, spike2lm)


if __name__ == '__main__':
    main()
