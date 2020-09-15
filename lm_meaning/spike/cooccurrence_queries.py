import argparse
from collections import defaultdict

import pandas as pd
import requests

import wandb

from lm_meaning.spike.utils import get_relations_data, dump_json

WIKIPEDIA_URL = "https://spike.pubmed-phrase-support.apps.allenai.org/api/3/search/query"
WIKIPEDIA_BASE_URL = "https://spike.pubmed-phrase-support.apps.allenai.org"


def log_wandb(args):
    pattern = args.data_file.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
    )

    wandb.init(
        name=f'{pattern}_spike_cooccurrences',
        project="memorization",
        tags=["spike", pattern, 'cooccurrences'],
        config=config,
    )


def get_tsv_url(response: requests.models.Response, base_url) -> str:
    tsv_location = response.headers["tsv-location"]
    tsv_url = base_url + tsv_location + "?sentence_text=True&capture_indices=True&sentence_id=True"
    return tsv_url


def perform_query(query: str, dataset_name: str = "pubmed", query_type: str = "syntactic") -> pd.DataFrame:
    template = """{{
    "queries": {{"{query_type}": "{query_content}"}},
    "data_set_name": "{dataset_name}"
    }}"""

    query = template.format(query_content=query, dataset_name=dataset_name, query_type=query_type)
    headers = {'content-type': 'application/json'}

    url, base_url = WIKIPEDIA_URL, WIKIPEDIA_BASE_URL

    response = requests.post(url, data=query.encode('utf-8'), headers=headers)

    tsv_url = get_tsv_url(response, base_url=base_url)
    # print(tsv_url)
    df = pd.read_csv(tsv_url, sep="\t")

    # df = df[df['paragraph_text'].notnull()]

    return df


def construct_query(subjects, objects):
    enclosed_objs = ['`' + x + '`' for x in objects]
    enclosed_subjs = ['`' + x + '`' for x in subjects]

    query_str = "subject:{} object:{}"
    packed_query = query_str.format('|'.join(enclosed_subjs), '|'.join(enclosed_objs))
    return packed_query


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-data_file", "--data_file", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/trex/data/TREx/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/output/spike_results/cooccurrences/P449"
                               ".json")

    args = parse.parse_args()

    log_wandb(args)

    relations = get_relations_data(args.data_file)

    obj_dic = defaultdict(list)
    for row in relations:
        obj_dic[row['obj_label']].append(row['sub_label'])

    all_subjects = list(dict.fromkeys([item for sublist in obj_dic.values() for item in sublist]))
    all_objects = list(obj_dic.keys())
    # all_subjects = [x.replace('&', '\&') for x in all_subjects]
    # TODO - There is some bug in spike with the & token.
    print(len(all_subjects))
    all_subjects = [x for x in all_subjects if '&' not in x]
    print(len(all_subjects))
    query_match = construct_query(all_subjects, all_objects)
    # query_match = "subject:`Barack Obama`|`Joe Biden` object:`Hawaii`|`California`"

    dataset_name = "wiki"
    query_type = "boolean"

    # print(query_match)
    df_results = perform_query(query_match, dataset_name, query_type)

    subj_obj_counts = df_results.groupby(['subject', 'object']).size().to_dict()

    print('total number of objects found with all queries', sum(subj_obj_counts.values()))
    wandb.run.summary['total_occurrences'] = sum(subj_obj_counts.values())
    wandb.run.summary['n_subjects'] = len(all_subjects)
    wandb.run.summary['n_objects'] = len(all_objects)

    subj_obj_counts_dic = {}
    for k, v in subj_obj_counts.items():
        subj_obj_counts_dic['_SEP_'.join(k)] = v
    dump_json(subj_obj_counts_dic, args.spike_results)


if __name__ == '__main__':
    main()
