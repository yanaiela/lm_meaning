import argparse
from collections import defaultdict

from spike.search.queries.q import StructuredSearchQuery, TokenSearchQuery
from spike.annotators.annotator_service import Annotator
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.integration.odinson.common import OdinsonContinuationToken
from requests.exceptions import RequestException
import numpy as np
from typing import Iterator, Optional
from tqdm import tqdm

import wandb

from lm_meaning.spike.utils import get_spike_objects, get_relations_data, dump_json


def log_wandb(args):
    pattern = args.spike_patterns.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
        token_query=args.token_query,
    )

    wandb.init(
        entity='consistency',
        name=f'{pattern}_spike_preference',
        project="memorization",
        tags=["spike", pattern, 'preference'],
        config=config,
    )


def get_syntactic_results(engine: MatchEngine, annotator: Annotator, spike_query: str,
                          continuation: Optional[OdinsonContinuationToken] = None) -> Iterator[SearchMatch]:
    search_query = StructuredSearchQuery(spike_query, annotator=annotator)
    query_match = engine.match(search_query, continuation=continuation)
    return query_match


def get_token_results(engine: MatchEngine, annotator: Annotator, spike_query: str,
                      continuation: Optional[OdinsonContinuationToken] = None) -> Iterator[SearchMatch]:
    search_query = TokenSearchQuery(spike_query)
    query_match = engine.match(search_query, continuation=continuation)
    return query_match


def construct_syntactic_spike_query(pattern: str) -> str:
    spike_pattern = pattern.replace('[X]', ' [X] ').replace('[Y]', ' [Y] ').strip()
    tokens = spike_pattern.split()
    spike_tokens = []
    for i in range(len(tokens)):
        if tokens[i] == '[Y]':
            spike_tokens.append('<>object:object')

        elif tokens[i] == '.':
            continue
        elif tokens[i] == '[X]':
            spike_tokens.append('<>subject:subject')
        else:
            spike_tokens.append(f'${tokens[i]}')
    return ' '.join(spike_tokens)


def construct_token_spike_query(pattern: str) -> str:
    spike_pattern = pattern.replace('[X]', ' [X] ').replace('[Y]', ' [Y] ')
    spike_pattern = spike_pattern.replace('[X]', '*').strip()
    tokens = spike_pattern.split()
    spike_tokens = []
    for i in range(len(tokens)):
        if tokens[i] == '[Y]':
            # capturing a single token for the object
            spike_tokens.append('object:*')

        elif tokens[i] == '.':
            continue
        else:
            spike_tokens.append(f'{tokens[i]}')

    return ' '.join(spike_tokens)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="pattern file",
                       default="data/pattern_data/parsed/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="data/output/spike_results/preferences/P449.json")
    parse.add_argument("-token_query", "--token_query", action='store_true',
                       help="Use syntactic search (by default, when False, or the token query, when True)")

    args = parse.parse_args()
    log_wandb(args)

    patterns = [x['pattern'] for x in get_relations_data(args.spike_patterns)]

    if args.token_query:
        construct_query = construct_token_spike_query
        get_results = get_token_results
    else:
        construct_query = construct_syntactic_spike_query
        get_results = get_syntactic_results

    objects_per_pattern = {}
    for pattern in patterns:
        spike_query = construct_query(pattern)
        print('spike query:', spike_query)
        wandb.run.summary['pattern'] = pattern
        wandb.run.summary['spike_query'] = spike_query

        spike_engine, spike_annotator = get_spike_objects()

        obj_counts = defaultdict(int)

        query_match = get_results(spike_engine, spike_annotator, spike_query)
        more_results = True
        continuation_token = None

        while more_results:
            try:
                for match in tqdm(query_match):
                    if match is None:
                        break
                    continuation_token = match.continuation_token

                    # adding all words from the referred relation (which can take the object role)
                    for word_i in range(match.captures['object'].first, match.captures['object'].last + 1):
                        obj = match.sentence.words[word_i]
                        obj_counts[obj] += 1
                more_results = False
            except (ConnectionResetError, RequestException):
                query_match = get_results(spike_engine, spike_annotator, spike_query, continuation_token)
        objects_per_pattern[pattern] = obj_counts

    n_patterns = len(objects_per_pattern)
    n_objects = []
    for row in objects_per_pattern.values():
        n_objects.append(len(row))
    wandb.run.summary['n_pattern'] = np.mean(n_patterns)
    wandb.run.summary['avg. objects per pattern'] = np.mean(n_objects)

    dump_json(objects_per_pattern, args.spike_results)


if __name__ == '__main__':
    main()
