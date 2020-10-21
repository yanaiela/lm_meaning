import argparse
import glob
import json
from glob import glob

import numpy as np
from scipy.stats import spearmanr


def construct_predictions(relation_predictions):
    data = relation_predictions['data']
    preds = relation_predictions['predictions']

    tokens_distribution = []
    for data_row, pred_row in zip(data, preds):
        tokens_window = []
        if pred_row.__class__ is not list:
            print('why not?')
        for entry in pred_row:
            tokens_window.append(entry['token_str'])
        tokens_distribution.append(tokens_window)
    return tokens_distribution


def order_rank_score(biased_distribution, subj_obj_distribution):
    filtered_biased_distribution = [x for x in biased_distribution if x in subj_obj_distribution]

    tokens2id = {k: v for v, k in (enumerate(filtered_biased_distribution))}

    # dmp = dmp_module.diff_match_patch()
    # diff = dmp.diff_main(''.join([str(tokens2id[x]) for x in filtered_biased_distribution]),
    #                      ''.join([str(tokens2id[x]) for x in subj_obj_distribution]))
    # levenshtein_diff = dmp.diff_levenshtein(diff)
    # diff = nltk.edit_distance(filtered_biased_distribution, subj_obj_distribution)
    diff = spearmanr(np.array([tokens2id[x] for x in filtered_biased_distribution]),
                     np.array([tokens2id[x] for x in subj_obj_distribution])) \
        .correlation
    return diff


def window_match_score_old(biased_distribution, subj_obj_distribution):
    window_size = len(subj_obj_distribution)
    match_scores = []
    for i in range(0, len(biased_distribution) - window_size, 1):
        score = sum([biased_distribution[i + j] == subj_obj_distribution[j] for j in range(window_size)])
        match_scores.append(score)

    return max(match_scores)


def window_match_score(biased_distribution, subj_obj_distribution):
    window_size = len(subj_obj_distribution)
    biased_conv = np.zeros(len(biased_distribution))
    filter_conv = np.ones(window_size)

    for obj in subj_obj_distribution:
        ind = biased_distribution.index(obj)
        biased_conv[ind] = 1

    conv = np.convolve(filter_conv, biased_conv, 'valid')

    # max_match = 0
    # for ind, i_conv in enumerate(conv):
    #     if i_conv > 0:
    #         reduced_biased_dist = biased_distribution[ind: ind + window_size]
    #         match = sum([reduced_biased_dist[j] == subj_obj_distribution[j] for j in range(window_size)])
    #         if match > max_match:
    #             max_match = match
    #
    # return max_match
    return int(max(conv))


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-lm_preference_file", "--lm_preference_file", type=str, help="lm preference file",
                       default="data/preference_bias/bias.json")
    parse.add_argument("-data_path", "--data_path", type=str, help="lm prediction file",
                       default="data/output/predictions_lm/lama/")

    args = parse.parse_args()

    with open(args.lm_preference_file, 'r') as f:
        preference_bias_data = json.load(f)

    for file_name in glob(args.data_path + '/*'):
        # print(file_name)
        with open(file_name, 'r') as f:
            lm_preds = json.load(f)

        orig_pattern = list(lm_preds.keys())[0]

        relation_id = file_name.split('/')[-1].split('_')[0]
        # if relation_id in ['P1001']:
        if relation_id in ['P37', 'P413', 'P364']:
            continue

        preference_bias = preference_bias_data[relation_id]

        relation_predictions = lm_preds[orig_pattern]

        object_distributions = construct_predictions(relation_predictions)

        per_bucket_rank = []
        per_bucket_match = []
        for i in ([5, 10, 15, 20, 30, 50]):
            avg_ranks = []
            avg_window_match = []
            for obj_dist in object_distributions:
                rank = order_rank_score(preference_bias, obj_dist[:i])
                avg_ranks.append(rank)
                # window_match = window_match_score(preference_bias, obj_dist[:i])
                # avg_window_match.append(window_match)
            # print(i, sum(avg) / len(avg))
            per_bucket_rank.append(sum(avg_ranks) / len(avg_ranks))
            # per_bucket_match.append(sum(avg_window_match) / len(avg_window_match))
            # print(per_bucket_match)
        print(relation_id, end=' & ')
        for ind, x in enumerate(per_bucket_rank):
        # for x in per_bucket_match:
            if ind == len(per_bucket_rank) - 1:
                print("{:.2f}".format(x), end=' ')
            else:
                print("{:.2f}".format(x), end=' & ')
        print('\\\\')
        # break


if __name__ == '__main__':
    main()
