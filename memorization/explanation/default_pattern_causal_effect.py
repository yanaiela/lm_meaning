"""

"""

import argparse
import pandas as pd
from collections import defaultdict
from glob import glob
from tqdm.auto import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
from memorization.explanation.causal_effect_utils import read_data, log_wandb
import wandb


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
        for obj, count in dic.items():
            counts[pattern][obj] = count

    most_common = {}
    for pattern, obj_dict in counts.items():
        desc = OrderedDict(sorted(obj_dict.items(),
                                  key=lambda kv: kv[1], reverse=True))
        second = None
        if len(desc) > 1:
            second = list(desc)[1]
        first = list(desc)[0]
        most_common[pattern] = (first, second, counts[pattern][first])
    return most_common


def parse_data_most_common(trex, data, raw_patterns, memorization, spike2pat):
    trex_dic = defaultdict(dict)
    trex_table = []

    for row in trex:
        subj = row['sub_label']
        obj = row['obj_label']
        trex_dic[subj] = obj
        trex_table.append([subj, obj])

    trex_df = pd.DataFrame(trex_table, columns=['subject', 'object'])
    subjs = trex_df['subject'].unique()

    most_common = get_default_object(data)
    most_common_objects = set()
    for tup in most_common.values():
        most_common_objects.add(tup[0])
        most_common_objects.add(tup[1])

    cooccurrences_table = []
    for subj in subjs:
        for pat in raw_patterns:
            if pat in most_common and most_common[pat][2] > 5:
                for obj in most_common[pat][:2]:
                    mem = False
                    if obj in memorization and subj in memorization[obj]:
                        for mem_pat in memorization[obj][subj]:
                            if mem_pat in spike2pat and spike2pat[mem_pat] == pat:
                                mem = True
                    if obj == None or obj != most_common[pat][0]:
                        is_most_common = False
                    else:
                        is_most_common = True
                    cooccurrences_table.append([subj, obj, pat, is_most_common, mem, 'born-in'])

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
        x_df = p_df[p_df['def-object'].values == def_obj]
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
    parse.add_argument("-n", "--num")
    parse.add_argument('--random_weights', default=False, type=lambda x: (str(x).lower() == 'true'),
                       help="randomly initialize the models' weights")

    args = parse.parse_args()

    log_wandb(args, 'pattern-object-cooccurrence')

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

        filt_data = filter_objects(obj_preference_data, trex)
        raw_patterns = list(spike2pat.values())

        df = parse_data_most_common(trex, filt_data, raw_patterns, memorization, spike2pat)
        para_pred_df = patterns_parse(paraphrase_preds)
        #import pdb; pdb.set_trace()

        # Merging the paraphrase predictions with the KB entities, while keeping the KB values the same, and duplicating
        #  each one of these rows based on the amount of paraphrases for this relation
        df_merge = df.merge(para_pred_df, how='left', on=['subject', 'pattern'])
        df_merge = df_merge.drop('object_y', axis=1).drop_duplicates().rename(columns={'object_x': 'object'})

        final_df.append(df_merge)

    wandb.run.summary['n. patterns'] = len(final_df)
    df = pd.concat(final_df)
    # In the case of the Roberta models is used, removing the spare space
    df['prediction'] = df.apply(lambda x: x['prediction'].strip(), axis=1)

    # lower casing in the case of the multiberts models, as they trained the uncased version of bert
    if 'google' in args.model:
        df['object'] = df.apply(lambda x: x['object'].lower(), axis=1)
    elif 'albert' in args.model:
        tok = AutoTokenizer.from_pretrained(args.model)
        df['object'] = df.apply(lambda x: tok.tokenize(x['object'])[0], axis=1)
    df['pred_def'] = df['prediction'] == df['object']

    res_treatment = estimate_p(df, True)
    res_control = estimate_p(df, False)
    wandb.run.summary['E[1]'] = res_treatment
    wandb.run.summary['E[0]'] = res_control
    wandb.run.summary['ATE'] = res_treatment - res_control


if __name__ == '__main__':
    main()
