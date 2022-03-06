"""

"""

import argparse
import pandas as pd
import json
from collections import defaultdict
import itertools
from glob import glob
from tqdm.auto import tqdm


def read_from_files(pattern: str, model: str):
    with open(f'data/output/spike_results/cooccurrences/{pattern}.json', 'r') as f:
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

    return data, trex, paraphrase_preds, memorization, patterns


def parse_data(trex, data, raw_patterns, memorization, spike2pat):
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

    counts = defaultdict(dict)
    for k, v in data.items():
        subj, obj = k.split('_SEP_')
        counts[subj][obj] = v

    cooccurrences_table = []
    for subj, obj in tqdm(itertools.product(subjs, objs)):
        for pattern in raw_patterns:

            mem = False
            if obj in memorization and subj in memorization[obj]:
                for mem_pat in memorization[obj][subj]:
                    if mem_pat in spike2pat and spike2pat[mem_pat] == pattern:
                        mem = True
            in_kbt = obj in trex_dic[subj]
            cooccurrences_table.append(
                [subj, obj, counts.get(subj, {'': 0}).get(obj, 0), pattern, mem, in_kbt, 'born-in'])

    df = pd.DataFrame(cooccurrences_table,
                      columns=['subject', 'object', 'count', 'pattern', 'memorization', 'in_kb', 'relation'])
    return df


def patterns_parse(paraphrase_preds):
    # Getting the paraphrase predictions
    paraphrase_preds_tab = []
    for pattern, vals in paraphrase_preds.items():
        for subj, (pred, obj) in vals.items():
            paraphrase_preds_tab.append([subj, obj, pattern, pred])

    para_pred_df = pd.DataFrame(paraphrase_preds_tab, columns=['subject', 'object', 'pattern', 'prediction'])
    return para_pred_df


def count_bins(row):
    count = row['count']
    if count <= 1:
        return 'xs'
    elif count <= 10:
        return 's'
    elif count <= 100:
        return 'm'
    elif count <= 1000:
        return 'l'
    else:
        return 'xl'


def estimate_p(df, bin_count, patterns, memorized=True):
    total_p = 0
    for count in tqdm(bin_count):
        for pattern in patterns:
            for in_kbt in [True, False]:
                p_df = df[(df['bin_count'] == count) & (df['pattern'] == pattern) & (df['in_kb'] == in_kbt)]
                x_df = p_df[p_df['memorization'] == memorized]
                p = len(p_df) / len(df)
                if len(x_df) != 0:
                    x_p = sum(x_df['pred_memorized'].values == True) / len(x_df)
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
    #for f in tqdm(glob(r'data/output/unpatterns/*_bert-large-cased.jsonl')):
    for f in tqdm(glob(r'data/output/predictions_lm/bert_lama_unpatterns/*_bert-large-cased.json')):
        pattern = f.split('unpatterns/')[1].split('_')[0]
        print(pattern)
        # if using a single pattern, discontinuing for other patterns
        if args.pattern != 'all' and args.pattern != pattern:
            continue

        data, trex, paraphrase_preds, memorization, patterns = read_from_files(pattern, args.model)

        spike2pat = {}

        for row in patterns:
            # filter only to paraphrases
            if row['pattern'] not in paraphrase_preds.keys(): continue
            spike2pat[row['spike_query']] = row['pattern']

        raw_patterns = list(spike2pat.values())
        df = parse_data(trex, data, raw_patterns, memorization, spike2pat)
        para_pred_df = patterns_parse(paraphrase_preds)

        df_merge = df.merge(para_pred_df, how='left', on=['subject', 'pattern'])
        df_merge = df_merge.drop('object_y', axis=1).drop_duplicates().rename(columns={'object_x': 'object'})

        df = df_merge

        df['bin_count'] = df.apply(lambda row: count_bins(row), axis=1)
        df['pred_memorized'] = df['object'] == df['prediction']

        filtered_rows = []
        for _, row in df[df['memorization'] == True].iterrows():

            try:
                negative = df[(df['subject'] == row['subject']) & (df['object'] == row['object']) & \
                              (df['bin_count'] == row['bin_count']) & (df['memorization'] == False)]\
                    .sample(1, random_state=1)
                filtered_rows.append(row.tolist())
                filtered_rows.append(negative.iloc[0].tolist())
            except:
                continue

        sampled = pd.DataFrame(filtered_rows, columns=df.columns)

        final_df.append(sampled)

    print(len(final_df))
    df = pd.concat(final_df)

    bin_counts = df['bin_count'].unique()
    patterns = df['pattern'].unique()

    res_treatment = estimate_p(df, bin_counts, patterns, True)
    res_control = estimate_p(df, bin_counts, patterns, False)
    print(res_treatment, res_control)
    print(res_treatment - res_control)


if __name__ == '__main__':
    main()
