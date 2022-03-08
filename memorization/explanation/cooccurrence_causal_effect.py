"""
Updated version, correct for January 9th, 2022.
Containing the correct code for the updated graph,
 where we need to control for the co-occurrence variable
 and not the in-kb

"""

import argparse
import pandas as pd
from collections import defaultdict
from glob import glob
from tqdm.auto import tqdm
from collections import OrderedDict
from memorization.explanation.causal_effect_utils import read_data, count_bins, log_wandb
import wandb


def get_most_cooccurring(data):
    counts = defaultdict(dict)
    for k, v in data.items():
        subj, obj = k.split('_SEP_')
        counts[subj][obj] = v

    most_common = {}
    for subj, obj_dict in counts.items():
        desc = OrderedDict(sorted(obj_dict.items(),
                                      key=lambda kv: kv[1], reverse=True))
        most_common[subj] = (list(desc)[:2], list(desc.values())[:2])
    return most_common


def parse_data_most_common(trex, data):
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

    most_common = get_most_cooccurring(data)

    cooccurrences_table = []
    for subj in subjs:

        second_cooc = None
        if subj in most_common:
            most_cooc = most_common[subj][0][0]
            most_c = most_common[subj][1][0]

            if len(most_common[subj][0]) > 1:
                second_cooc = most_common[subj][0][1]
                second_c = most_common[subj][1][1]
            # else:

        if second_cooc is None:
            continue
        if most_cooc in trex_dic[subj]:
            cooccurrences_table.append([subj, most_cooc, most_c, True, True, 'born-in'])
        else:
            cooccurrences_table.append([subj, most_cooc, most_c, True, False, 'born-in'])

        if second_cooc in trex_dic[subj]:
            cooccurrences_table.append([subj, second_cooc, second_c, False, True, 'born-in'])
        else:
            cooccurrences_table.append([subj, second_cooc, second_c, False, False, 'born-in'])

    df = pd.DataFrame(cooccurrences_table, columns=['subject', 'object', 'count', 'most-common', 'in_kb', 'relation'])
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
            continue
        else:
            for subj, (pred, obj) in vals.items():
                unpatterns_tab.append([subj, obj, False, 'born-in', pattern, pred])

    unpatterns_df = pd.DataFrame(unpatterns_tab,
                                 columns=['subject', 'object', 'in_kb', 'relation', 'pattern', 'prediction'])
    return unpatterns_df


def estimate_p(df, bin_count, most_common=True):
    total_p = 0
    for count in tqdm(bin_count):
        p_df = df[df['bin_count'].values == count]
        x_df = p_df[p_df['most-common'].values == most_common]
        p = len(p_df) / len(df)
        if len(x_df) != 0:
            x_p = sum(x_df['pred_cooc'].values == True) / len(x_df)
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
    parse.add_argument("--random_weights", action='store_true', default=False, help="use random weights model")

    args = parse.parse_args()
    log_wandb(args, 'subject-object-cooccurrence')

    final_df = []
    for f in tqdm(glob(r'data/output/predictions_lm/bert_lama_unpatterns/*_bert-large-cased.json')):
        #print(f)
        pattern = f.split('unpatterns/')[1].split('_')[0]
        print(pattern)
        # if using a single pattern, discontinuing for other patterns
        if args.pattern != 'all' and args.pattern != pattern:
            continue

        co_occurrence_data, obj_preference_data, trex, paraphrase_preds, unparaphrase_preds, memorization, patterns = read_data(pattern, args.model, args.random_weights)

        df = parse_data_most_common(trex, co_occurrence_data)
        para_pred_df = patterns_parse(paraphrase_preds)
        unpatterns_df = unpatterns_parse(unparaphrase_preds)

        # Merging the paraphrase predictions with the KB entities, while keeping the KB values the same, and duplicating
        #  each one of these rows based on the amount of paraphrases for this relation
        df_merge = df.merge(para_pred_df, how='left', on=['subject'])
        df_merge = df_merge.drop('object_y', axis=1).drop_duplicates().rename(columns={'object_x': 'object'})
        unpattern_merge = df.merge(unpatterns_df, how='left', on=['subject'])
        unpattern_merge = unpattern_merge.drop(['object_y', 'in_kb_y', 'relation_y'], axis=1).drop_duplicates().rename(
            columns={'object_x': 'object', 'in_kb_x': 'in_kb', 'relation_x': 'relation'})
        df = pd.concat([df_merge, unpattern_merge])

        final_df.append(df)

    wandb.run.summary['n. patterns'] = len(final_df)
    df = pd.concat(final_df)

    # In the case of the Roberta models is used, removing the spare space
    df['prediction'] = df.apply(lambda x: x['prediction'].strip(), axis=1)

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

    df['bin_count'] = df.apply(lambda row: count_bins(row), axis=1)

    # In case using the google model (which is uncased), lower casing the objects
    if 'google' in args.model:
        df['object'] = df.apply(lambda x: x['object'].lower(), axis=1)
    # Are predictions correct
    df['pred_cooc'] = df['object'] == df['prediction']
    df['bin_cooccurrence'] = df['count'] == df['prediction']

    bin_counts = df['bin_count'].unique()

    res_treatment = estimate_p(df, bin_counts, True)
    res_control = estimate_p(df, bin_counts, False)
    wandb.run.summary['E[1]'] = res_treatment
    wandb.run.summary['E[0]'] = res_control
    wandb.run.summary['ATE'] = res_treatment - res_control


if __name__ == '__main__':
    main()
