import argparse
import pandas as pd
import json
from collections import defaultdict
import itertools
from joblib import Parallel, delayed

from tqdm.auto import tqdm


def read_from_files(pattern: str, model: str):
    with open(f'data/output/spike_results/cooccurrences/{pattern}.json', 'r') as f:
        data = json.load(f)

    with open(f'data/trex_lms_vocab/{pattern}.jsonl', 'r') as f:
        trex = f.readlines()
        trex = [json.loads(x.strip()) for x in trex]

    with open(f'data/output/predictions_lm/trex_lms_vocab/{pattern}_{model}.json', 'r') as f:
        paraphrase_preds = json.load(f)

    with open(f'data/output/unpatterns/{pattern}_{model}.jsonl', 'r') as f:
        unparaphrase_preds = json.load(f)

    return data, trex, paraphrase_preds, unparaphrase_preds


def parse_data(trex, data):
    trex_dic = defaultdict(dict)

    for row in trex:
        trex_dic[row['sub_label']][row['obj_label']] = True

    # Registring the data from the KB
    cooccurrences_table = []

    for row in trex:
        subj = row['sub_label']
        obj = row['obj_label']
        cooc = data.get('_SEP_'.join([subj, obj]), 0)
        cooccurrences_table.append([subj, obj, cooc, True, 'born-in'])

    df = pd.DataFrame(cooccurrences_table, columns=['subject', 'object', 'cooccurrence', 'in_kb', 'relation'])
    return df


def patterns_parse(paraphrase_preds):
    # Getting the paraphrase predictions
    paraphrase_preds_tab = []
    for pattern, vals in paraphrase_preds.items():
        for subj, (pred, obj) in vals.items():
            paraphrase_preds_tab.append([subj, obj, pattern, pred])

    para_pred_df = pd.DataFrame(paraphrase_preds_tab, columns=['subject', 'object', 'pattern', 'prediction'])
    return para_pred_df


def unpatterns_parse(unparaphrase_preds):
    # Populating the predictions for a relations that are unlikely to be true (and thus not appear in the KB)
    unpatterns_tab = []
    for ind, (pattern, vals) in enumerate(unparaphrase_preds.items()):
        # first pattern in these files is the original (correct) pattern describing the relation
        if ind == 0:
            orig_pattern = pattern
        else:
            for data, preds in zip(vals['data'], vals['predictions']):
                unpatterns_tab.append(
                    [data['sub_label'], data['obj_label'], False, 'born-in', pattern, preds[0]['token_str']])

    unpatterns_df = pd.DataFrame(unpatterns_tab,
                                 columns=['subject', 'object', 'in_kb', 'relation', 'pattern', 'prediction'])
    return unpatterns_df


# simple calculation function (for interpretability)
def estimate_p(df, subjects, objects, so_in_kb, cooc):
    total_p = 0
    for (s, o, so_kb, c) in tqdm(itertools.product(subjects, objects, so_in_kb, cooc)):
        p_df = df[(df['subject'] == s) & (df['object'] == o) & (df['in_kb'] == so_kb)]
        x_df = p_df[p_df['bin_cooccurrence'] == c]
        p = len(p_df) / len(df)
        if len(x_df) != 0:
            x_p = sum(x_df['pred_cooc'] is True) / len(x_df)
        else:
            x_p = 0
        total_p += p * x_p


# fast calc, multi-process
def batch_estimate(s_df, len_df, objects, so_in_kb, treatment: bool):
    t = 0.
    for o in objects:
        for so_kb in so_in_kb:
            p_df = s_df[(s_df['object'].values == o) & (s_df['in_kb'].values == so_kb)]
            p = len(p_df) / len_df
            if p == 0.:
                continue
            # for c in cooc:
            x_df = p_df[p_df['bin_cooccurrence'].values == treatment]

            if len(x_df) != 0:
                x_p = sum(x_df['pred_cooc'].values is True) / len(x_df)
            else:
                # x_p = 0
                continue
            t += p * x_p
    return t


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-p", "--pattern", type=str, help="pattern id",
                       default="P449")
    parse.add_argument("-m", "--model", type=str, help="model",
                       default="bert-large-cased")

    args = parse.parse_args()

    data, trex, paraphrase_preds, unparaphrase_preds = read_from_files(args.pattern, args.model)

    df = parse_data(trex, data)
    para_pred_df = patterns_parse(paraphrase_preds)
    unpatterns_df = unpatterns_parse(unparaphrase_preds)

    # Merging the paraphrase predictions with the KB entities, while keeping the KB values the same, and duplicating
    #  each one of these rows based on the amount of paraphrases for this relation
    df_merge = df.merge(para_pred_df, how='left', on=['subject', 'object'])

    subj_obj_cooc = df_merge[['subject', 'object', 'cooccurrence']].drop_duplicates()

    # Similarly to the paraphrases, merging the "unpattern" data
    unpatterns_df = unpatterns_df.merge(subj_obj_cooc, on=['subject', 'object'])
    df = pd.concat([df_merge, unpatterns_df])

    # Are predictions correct
    df['pred_cooc'] = df['object'] == df['prediction']

    # Counting the number of times each subject and object appeared in the KB (based on their mutual cooccurrences)
    subj_obj = df[['subject', 'object']].drop_duplicates()
    obj_count = subj_obj['object'].value_counts()
    sub_count = subj_obj['subject'].value_counts()

    sub_count = pd.DataFrame(sub_count, columns=['subject']).reset_index().rename(
        {'subject': 'subject_c', 'index': 'subject'}, axis=1)
    obj_count = pd.DataFrame(obj_count, columns=['object']).reset_index().rename(
        {'object': 'object_c', 'index': 'object'}, axis=1)

    # Adding the subj/obj counts to the data
    df = df.merge(obj_count, on=['object']).merge(sub_count, on=['subject'])

    df['bin_cooccurrence'] = df['cooccurrence'] > 0

    subjects = df['subject'].unique()
    objects = df['object'].unique()
    so_in_kb = df['in_kb'].unique()
    cooc = df['bin_cooccurrence'].unique()

    len_df = len(df)
    res_treatment = Parallel(n_jobs=10)(
        delayed(batch_estimate)(df[(df['subject'].values == s)], len_df, objects, so_in_kb, True) for s in subjects)
    res_control = Parallel(n_jobs=10)(
        delayed(batch_estimate)(df[(df['subject'].values == s)], len_df, objects, so_in_kb, False) for s in subjects)
    print(sum(res_treatment) - sum(res_control))


if __name__ == '__main__':
    main()
