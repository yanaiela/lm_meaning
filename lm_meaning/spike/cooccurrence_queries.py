import argparse
from collections import defaultdict
from requests.exceptions import RequestException
import pandas as pd
import requests
from tqdm import tqdm
import wandb
from spike.search.queries.q import BooleanSearchQuery
from typing import List, Iterator, Optional, Dict, Tuple
from lm_meaning.spike.utils import get_relations_data, dump_json, get_spike_objects, enclose_entities
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.integration.odinson.common import OdinsonContinuationToken
from spike.spacywrapper.annotator import SpacyAnnotator


WIKIPEDIA_URL = "https://spike.staging.apps.allenai.org/api/3/search/query"
WIKIPEDIA_BASE_URL = "https://spike.staging.apps.allenai.org"


def log_wandb(args):
    pattern = args.data_file.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
    )

    wandb.init(
        entity='consistency',
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
    df = pd.read_csv(tsv_url, sep="\t")

    # df = df[df['paragraph_text'].notnull()]

    return df


def construct_query(subjects: List[str], objects: List[str], engine: MatchEngine,
                    continuation: Optional[OdinsonContinuationToken] = None) -> Iterator[SearchMatch]:
    enclosed_objs = ['`' + x + '`' for x in objects]
    enclosed_subjs = ['`' + x + '`' for x in subjects]

    query_str = "subject:{} object:{}"
    packed_query = query_str.format('|'.join(enclosed_subjs), '|'.join(enclosed_objs))

    search_query = BooleanSearchQuery(packed_query)
    query_match = engine.match(search_query, continuation)

    return query_match


def prepare_data(kb_tuples: List[Dict]) -> Tuple[List[str], List[str]]:
    all_objects = []
    all_subjects = []
    for row in kb_tuples:
        all_objects.append(row['obj_label'])
        all_subjects.append(row['sub_label'])

    all_objects = list(set(all_objects))
    all_subjects = list(set(all_subjects))

    spacy_annotator = SpacyAnnotator.from_config("en.json")

    # tokenize the subjects to the way spacy tokenizes them
    all_subjects = [enclose_entities(spacy_annotator, subj) for subj in all_subjects]
    return all_subjects, all_objects


def get_cooccurrences(subjects: List[str], objects: List[str], spike_engine) -> Dict[str, int]:
    query_match = construct_query(subjects, objects, spike_engine)

    subj_obj_counts_dic = defaultdict(int)
    more_results = True
    continuation_token = None
    while more_results:
        try:
            for match in tqdm(query_match):
                continuation_token = match.continuation_token
                obj = ' '.join(match.sentence.words[match.captures['object'].first: match.captures['object'].last + 1])
                subj = ' '.join(
                    match.sentence.words[match.captures['subject'].first: match.captures['subject'].last + 1])
                subj_obj_counts_dic['_SEP_'.join([subj, obj])] += 1
            more_results = False
        except (ConnectionResetError, RequestException):
            query_match = construct_query(subjects, objects, spike_engine, continuation_token)
        if continuation_token is None:
            more_results = False
    return subj_obj_counts_dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-data_file", "--data_file", type=str,
                       help="input trex file, containing the subject/object tuples",
                       default="data/trex_lms_vocab/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="data/output/spike_results/cooccurrences/P449.json")

    args = parse.parse_args()

    log_wandb(args)

    spike_engine, _ = get_spike_objects()

    relations = get_relations_data(args.data_file)

    all_subjects, all_objects = prepare_data(relations)

    subj_obj_counts_dic = get_cooccurrences(all_subjects, all_objects, spike_engine)

    print('total number of objects found with all queries', sum(subj_obj_counts_dic.values()))
    wandb.run.summary['total_occurrences'] = sum(subj_obj_counts_dic.values())
    wandb.run.summary['n_subjects'] = len(all_subjects)
    wandb.run.summary['n_objects'] = len(all_objects)

    dump_json(subj_obj_counts_dic, args.spike_results)


if __name__ == '__main__':
    main()
