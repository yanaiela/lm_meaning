import argparse
from collections import defaultdict
from typing import Iterator, Optional, List
from requests.exceptions import RequestException
from spike.annotators.annotator_service import Annotator
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.search.queries.q import StructuredSearchQuery
from spike.integration.odinson.common import OdinsonContinuationToken

from tqdm import tqdm
import wandb
from memorization.spike import get_spike_objects, get_relations_data, dump_json


def log_wandb(args):
    pattern = args.spike_patterns.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
    )

    wandb.init(
        entity='consistency',
        name=f'{pattern}_wiki_patterns',
        project="memorization",
        tags=["spike", pattern, 'wiki_patterns'],
        config=config,
    )


def construct_query(engine: MatchEngine, annotator: Annotator, query_str: str,
                    continuation: Optional[OdinsonContinuationToken] = None) -> Iterator[SearchMatch]:

    search_query = StructuredSearchQuery(query_str, annotator=annotator)
    query_match = engine.match(search_query, continuation=continuation)
    return query_match


def count_patterns(patterns: List[str], spike_engine, spike_annotator):
    data_dic = defaultdict(int)
    # The following code is slightly complicated due to the continuation token option, which allows to reset the
    #  iterator which is connected to spike, to avoid connection issues.
    # If a connection error happen due to a spike connection, the query is reconstructed with the continuation token
    #  and resumed from there.
    for pattern in tqdm(patterns):
        more_results = True
        count = 0
        continuation_token = None
        query_pattern = pattern.replace('[w={}]', '')
        query_match = construct_query(spike_engine, spike_annotator, query_pattern)
        while more_results:
            try:
                for r in tqdm(query_match):
                    continuation_token = r.continuation_token
                    count += 1
                more_results = False
            except (ConnectionResetError, RequestException):
                query_match = construct_query(spike_engine, spike_annotator, query_pattern,
                                              continuation=continuation_token)
            if continuation_token is None:
                more_results = False
        data_dic[pattern] = count

    return data_dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="pattern file",
                       default="data/pattern_data/parsed/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="data/output/spike_results/pattern_counts/P449.json")

    args = parse.parse_args()
    log_wandb(args)

    spike_engine, spike_annotator = get_spike_objects()

    patterns = [x['spike_query'] for x in get_relations_data(args.spike_patterns)]

    data_dic = count_patterns(patterns, spike_engine, spike_annotator)

    for pattern, count in data_dic.items():
        print('pattern: {}. count: {}'.format(pattern, count))

    table = wandb.Table(columns=["Pattern", "Count"])
    for pattern, count in data_dic.items():
        table.add_data(pattern, count)
    wandb.log({"patterns_results": table})
    wandb.run.summary['n_patterns'] = len(patterns)
    dump_json(data_dic, args.spike_results)


if __name__ == '__main__':
    main()
