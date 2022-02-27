"""

"""

import argparse
import pandas as pd
import json
from collections import defaultdict
import itertools
from joblib import Parallel, delayed
from glob import glob
from tqdm.auto import tqdm
from collections import OrderedDict


def read_from_files(pattern: str, model: str):
    with open(f'data/output/spike_results/preferences/{pattern}.json', 'r') as f:
        data = json.load(f)

    with open(f'data/trex_lms_vocab/{pattern}.jsonl', 'r') as f:
        trex = f.readlines()
        trex = [json.loads(x.strip()) for x in trex]

    with open(f'data/output/predictions_lm/trex_lms_vocab/{pattern}_{model}.json', 'r') as f:
        paraphrase_preds = json.load(f)

    with open(f'data/output/predictions_lm/bert_lama_unpatterns/{pattern}_{model}.json', 'r') as f:
        unparaphrase_preds = json.load(f)

    with open(f'data/output/spike_results/paraphrases/{pattern}.json', 'r') as f:
        memorization = json.load(f)

    with open(f'data/pattern_data/parsed/{pattern}.jsonl') as f:
        patterns = [json.loads(x.strip()) for x in f.readlines()]

    return data, trex, paraphrase_preds, unparaphrase_preds, memorization, patterns


def filter_objects(data, trex):
    obj_set = set([row['obj_label'] for row in trex])

    filt_data = defaultdict(dict)
    for pat, dic in data.items():
        for obj, val in dic.items():
            if obj in obj_set:
                filt_data[pat][obj] = val
    return filt_data


def get_default_object(data):
    counts = defaultdict(dict)
    for pattern, dic in data.items():
        for subj, count in dic.items():
            counts[pattern][subj] = count

    most_common = {}
    for pattern, obj_dict in counts.items():
        desc = OrderedDict(sorted(obj_dict.items(),
                                      key=lambda kv: kv[1], reverse=True))
        second = None
        if len(desc) > 1:
            second = list(desc)[1]
        most_common[pattern] = (list(desc)[0], second)
    return most_common


def parse_data_most_common(trex, data, raw_patterns, memorization, spike2pat):
    trex_dic = defaultdict(dict)
    trex_table = []

    for row in trex:
        subj = row['sub_label']
        obj = row['obj_label']
        trex_dic[subj][obj] = True
        trex_table.append([subj, obj])

    trex_df = pd.DataFrame(trex_table, columns=['subject', 'object'])
    subjs = trex_df['subject'].unique()
    objs = trex_df['object'].unique()

    most_common = get_default_object(data)
    most_common_objects = set()
    for tup in most_common.values():
        most_common_objects.add(tup[0])
        most_common_objects.add(tup[1])
    #     print(most_common)
    #     print(most_common_objects)

    cooccurrences_table = []
    #     for subj, obj in tqdm(itertools.product(subjs, objs)):
    for subj in subjs:
        for obj in objs:
            #             if obj not in trex_dic[subj] and obj not in most_common_objects: continue
            if obj not in trex_dic[subj]: continue
            for pat in raw_patterns:
                #             print(subj, pat, pat in most_common)
                #             print(trex_dic[subj])
                if pat in most_common and obj == most_common[pat][0]:
                    #                     print('hi')
                    def_obj = most_common[pat][0]
                else:
                    def_obj = 'None'

                mem = False
                if obj in memorization and subj in memorization[obj]:
                    for mem_pat in memorization[obj][subj]:
                        if mem_pat in spike2pat and spike2pat[mem_pat] == pat:
                            mem = True

                #                 if def_obj in trex_dic[subj]:
                cooccurrences_table.append([subj, obj, pat, def_obj, mem, 'born-in'])

    #                 if pat in most_common and most_common[pat][1] is not None:
    #                     negative = most_common[pat][1]
    #                     cooccurrences_table.append([subj, negative, pat, def_obj, False, 'born-in'])
    #                 else:
    #                     cooccurrences_table.append([subj, obj, pat, def_obj, False, mem, 'born-in'])

    df = pd.DataFrame(cooccurrences_table,
                      columns=['subject', 'object', 'pattern', 'def-object', 'memorized', 'relation'])
    return df


def patterns_parse(paraphrase_preds):
    # Getting the paraphrase predictions
    paraphrase_preds_tab = []
    for pattern, vals in paraphrase_preds.items():
        for subj, (pred, obj) in vals.items():
            paraphrase_preds_tab.append([subj, obj, pattern, pred])

    para_pred_df = pd.DataFrame(paraphrase_preds_tab, columns=['subject', 'object', 'pattern', 'prediction'])
    return para_pred_df


def estimate_p(df, def_obj=True):
    total_p = 0
    for memorized in [True, False]:
        p_df = df[df['memorized'].values == memorized]
        x_df = p_df[p_df['is_def_obj'].values == def_obj]
        p = len(p_df) / len(df)
        if len(x_df) != 0:
            x_p = sum(x_df['pred_def'].values == True) / len(x_df)
        else:
            x_p = 0
        total_p += p * x_p
    return total_p


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-p", "--pattern", type=str, help="pattern id",
                       default="all")
    parse.add_argument("-m", "--model", type=str, help="model",
                       default="bert-large-cased")

    args = parse.parse_args()

    final_df = []
    for f in tqdm(glob(r'data/output/unpatterns/*_bert-large-cased.jsonl')):
        pattern = f.split('unpatterns/')[1].split('_')[0]
        print(pattern)
        # if using a single pattern, discontinuing for other patterns
        if args.pattern != 'all' and args.pattern != pattern:
            continue

        data, trex, paraphrase_preds, unparaphrase_preds, memorization, patterns = read_from_files(pattern,
                                                                                                   args.model)

        spike2pat = {}

        for row in patterns:
            # filter only to paraphrases
            if row['pattern'] not in paraphrase_preds.keys(): continue
            spike2pat[row['spike_query']] = row['pattern']

        filt_data = filter_objects(data, trex)
        raw_patterns = list(spike2pat.values())

        # df = parse_data(trex, data)
        df = parse_data_most_common(trex, filt_data, raw_patterns, memorization, spike2pat)
        para_pred_df = patterns_parse(paraphrase_preds)

        # Merging the paraphrase predictions with the KB entities, while keeping the KB values the same, and duplicating
        #  each one of these rows based on the amount of paraphrases for this relation
        # df_merge = df.merge(para_pred_df, how='left', on=['subject', 'object'])
        df_merge = df.merge(para_pred_df, how='left', on=['subject', 'object', 'pattern'])

        try:
            df = pd.concat([
                df_merge[(df_merge['def-object'] != 'None')],
                df_merge[(df_merge['def-object'] == 'None') & (df_merge['memorized'] == True)] \
                    .sample(df_merge[(df_merge['def-object'] != 'None')].memorized.value_counts()[True]),
                df_merge[(df_merge['def-object'] == 'None') & (df_merge['memorized'] == False)] \
                    .sample(df_merge[(df_merge['def-object'] != 'None')].memorized.value_counts()[False])
            ])
        except KeyError as e:
            continue
        except ValueError as e:
            continue

        final_df.append(df)

    print(len(final_df))
    df = pd.concat(final_df)

    df['pred_def'] = df['prediction'] == df['object']
    df['is_def_obj'] = df['object'] == df['def-object']

    res_treatment = estimate_p(df, True)
    res_control = estimate_p(df, False)
    print(res_treatment - res_control)


if __name__ == '__main__':
    main()
