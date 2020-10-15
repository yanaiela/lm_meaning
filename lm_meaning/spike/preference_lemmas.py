import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List

import pandas as pd
from pandas import DataFrame
import wandb
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_meaning.spike.preference_queries import get_token_results
from lm_meaning.spike.utils import get_spike_objects, get_relations_data, dump_json


def log_wandb(args):
    pattern = args.data_file.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
    )

    wandb.init(
        name=f'{pattern}_spike_lemma_preference',
        project="memorization",
        tags=["spike", pattern, 'preference'],
        config=config,
    )


def get_pattern_essentials(paraphrases_file: str) -> DataFrame:
    patterns_df = pd.read_csv(paraphrases_file, sep='\t')
    patterns_df = patterns_df.loc[patterns_df['LEMMA'] != '?']

    patterns = patterns_df['RULE'].tolist()
    lemmas = patterns_df['LEMMA'].tolist()

    filtered_patterns_data = []
    for pattern, lemma in zip(patterns, lemmas):
        lemma_split = lemma.replace('-', ' ')
        obj_ind = pattern.index('[Y]')
        if lemma_split in pattern:
            lemma_ind = pattern.index(lemma_split)

        else:
            seqMatch = SequenceMatcher(None, lemma_split, pattern)
            match = seqMatch.find_longest_match(0, len(lemma_split), 0, len(pattern))
            lemma_ind = match.a
        lemma_first = lemma_ind < obj_ind
        filtered_patterns_data.append([lemma_split, pattern, lemma_first])
    filtered_patterns_data = pd.DataFrame(filtered_patterns_data, columns=['lemma', 'pattern', 'lemma_first'])
    filtered_patterns_data = filtered_patterns_data.drop_duplicates(subset=['lemma', 'lemma_first'])
    return filtered_patterns_data


def construct_token_spike_query(lemma: List[str], lemma_first: bool, object_list: List[str]) -> str:
    spike_tokens = []
    lemma_tokens = []
    for w in lemma:
        lemma_tokens.append(f'lemma=`{w}`')
    if lemma_first:
        spike_tokens.extend(lemma_tokens)
        spike_tokens.append('...')
        spike_tokens.append('object:[{}]'.format('|'.join(object_list)))
    else:
        spike_tokens.append('object:[{}]'.format('|'.join(object_list)))
        spike_tokens.append('...')
        spike_tokens.extend(lemma_tokens)

    return ' '.join(spike_tokens)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-data_file", "--data_file", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/trex/data/TREx/P19.jsonl")
    parse.add_argument("-paraphrases_file", "--paraphrases_file", type=str, help="patterns file from LAMA",
                       default="data/pattern_data/P19.tsv")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/output/spike_results/lemma_preferences/"
                               "P19.json")

    args = parse.parse_args()
    log_wandb(args)

    relations = get_relations_data(args.data_file)
    objects = []
    for row in relations:
        objects.append(row['obj_label'])
    objects = list(set(objects))

    filtered_patterns_data = get_pattern_essentials(args.paraphrases_file)

    spike_engine, spike_annotator = get_spike_objects()
    patterns_lemmas_cooccurrences = {}

    for ind, row in filtered_patterns_data.iterrows():
        lemma = row['lemma']
        pattern = row['pattern']
        lemma_first = row['lemma_first']
        spike_query = construct_token_spike_query(lemma.split(), lemma_first, objects)

        print('spike query:', spike_query)

        obj_counts = defaultdict(int)

        query_match = get_token_results(spike_engine, spike_annotator, spike_query)
        more_results = True
        continuation_token = None

        while more_results:
            try:
                for match in tqdm(query_match):
                    if match is None:
                        break
                    continuation_token = match.continuation_token
                    obj = match.sentence.words[match.captures['object'].first]

                    obj_counts[obj] += 1
                more_results = False
            except (ConnectionResetError, RequestException) as connection_error:
                query_match = get_token_results(spike_engine, spike_annotator, spike_query, continuation_token)

        patterns_lemmas_cooccurrences['_SEP_'.join([pattern, lemma, str(lemma_first)])] = obj_counts

    distinct_lemmas_order = len(patterns_lemmas_cooccurrences)
    print('total number of lemmas/order: ', distinct_lemmas_order)

    wandb.run.summary['unique_lemmas_order'] = distinct_lemmas_order

    dump_json(patterns_lemmas_cooccurrences, args.spike_results)


if __name__ == '__main__':
    main()
