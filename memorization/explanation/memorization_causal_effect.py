"""

"""

import argparse
import pandas as pd
from collections import defaultdict
import itertools
from glob import glob
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from memorization.explanation.causal_effect_utils import read_data, count_bins, log_wandb
import wandb


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
    parse.add_argument('--random_weights', default=False, type=lambda x: (str(x).lower() == 'true'),
                       help="randomly initialize the models' weights")

    args = parse.parse_args()
    log_wandb(args, 'memorization')

    final_df = []
    for f in tqdm(glob(r'data/output/predictions_lm/bert_lama_unpatterns/*_bert-large-cased.json')):
        pattern = f.split('unpatterns/')[1].split('_')[0]
        # if using a single pattern, discontinuing for other patterns
        if args.pattern != 'all' and args.pattern != pattern:
            continue

        co_occurrence_data, obj_preference_data, trex, paraphrase_preds, unparaphrase_preds, memorization, patterns = read_data(
            pattern, args.model, args.random_weights)

        spike2pat = {}

        for row in patterns:
            # filter only to paraphrases
            if row['pattern'] not in paraphrase_preds.keys(): continue
            spike2pat[row['spike_query']] = row['pattern']

        raw_patterns = list(spike2pat.values())
        df = parse_data(trex, co_occurrence_data, raw_patterns, memorization, spike2pat)
        para_pred_df = patterns_parse(paraphrase_preds)

        df_merge = df.merge(para_pred_df, how='left', on=['subject', 'pattern'])
        df_merge = df_merge.drop('object_y', axis=1).drop_duplicates().rename(columns={'object_x': 'object'})

        df = df_merge

        df['bin_count'] = df.apply(lambda row: count_bins(row), axis=1)

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

    wandb.run.summary['n. patterns'] = len(final_df)
    df = pd.concat(final_df)

    # In the case of the Roberta models is used, removing the spare space
    df['prediction'] = df.apply(lambda x: x['prediction'].strip(), axis=1)

    if 'google' in args.model:
        df['object'] = df.apply(lambda x: x['object'].lower(), axis=1)
    elif 'albert' in args.model:
        tok = AutoTokenizer.from_pretrained(args.model)
        df['object'] = df.apply(lambda x: tok.tokenize(x['object'])[0], axis=1)
    df['pred_memorized'] = df['object'] == df['prediction']

    bin_counts = df['bin_count'].unique()
    patterns = df['pattern'].unique()

    res_treatment = estimate_p(df, bin_counts, patterns, True)
    res_control = estimate_p(df, bin_counts, patterns, False)
    wandb.run.summary['E[1]'] = res_treatment
    wandb.run.summary['E[0]'] = res_control
    wandb.run.summary['ATE'] = res_treatment - res_control


if __name__ == '__main__':
    main()
