import argparse
from collections import defaultdict

from spike.search.queries.q import StructuredSearchQuery
from spike.annotators.annotator_service import Annotator
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.integration.odinson.common import OdinsonContinuationToken
from requests.exceptions import RequestException

from typing import Iterator, Optional
from tqdm import tqdm

import wandb

from lm_meaning.spike.utils import get_spike_objects, get_relations_data, dump_json


def log_wandb(args):
    pattern = args.relation

    config = dict(
        pattern=pattern,
    )

    wandb.init(
        name=f'{pattern}_spike_preference',
        project="memorization",
        tags=["spike", pattern, 'preference'],
        config=config,
    )


def construct_query(engine: MatchEngine, annotator: Annotator, spike_query: str,
                    continuation: Optional[OdinsonContinuationToken] = None) -> Iterator[SearchMatch]:
    search_query = StructuredSearchQuery(spike_query, annotator=annotator)
    query_match = engine.match(search_query, continuation=continuation)
    return query_match


def construct_spike_query(pattern: str) -> str:
    spike_pattern = pattern.replace('[X]', 'subject:subject').strip()
    tokens = spike_pattern.split()
    spike_tokens = [tokens[0]]
    for i in range(1, len(tokens)):
        if tokens[i] == '[Y]':
            break

        # kinda hacky, but this deals with P449 (the "aired on" relation), which for some reason added
        #  this word, which makes the results much more sparse
        if tokens[i] == 'originally':
            continue
        spike_tokens.append(f'${tokens[i]}')
    # capturing a multi-expression (the '<>' prefix)
    spike_tokens.append('<>object:object')
    spike_tokens.append('.')
    return ' '.join(spike_tokens)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-relation", "--relation", type=str, help="relation id (e.g. P449)",
                       default="P449")
    parse.add_argument("-patterns_file", "--patterns_file", type=str, help="patterns file from LAMA",
                       default="data/trex/data/relations.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/output/spike_results/preferences/P449"
                               ".json")

    args = parse.parse_args()
    log_wandb(args)

    relation = args.relation

    patterns = [x['template'] for x in get_relations_data(args.patterns_file) if x['relation'] == relation]
    assert len(patterns) == 1, 'more than single relation that matches...'
    pattern = patterns[0]

    if not pattern.endswith('[Y] .'):
        print('skipping patterns that don\'t end with the object ([Y].)')
        return

    spike_query = construct_spike_query(pattern)
    print('spike query:', spike_query)
    wandb.run.summary['pattern'] = pattern
    wandb.run.summary['spike_query'] = spike_query

    spike_engine, spike_annotator = get_spike_objects()

    obj_counts = defaultdict(int)

    query_match = construct_query(spike_engine, spike_annotator, spike_query)
    more_results = True
    continuation_token = None

    while more_results:
        try:
            for match in tqdm(query_match):
                if match is None:
                    break
                continuation_token = match.continuation_token

                # adding all words from the referred relation (which can take the object role)
                for word_i in range(match.captures['object'].first, match.captures['object'].first + 1):
                    obj = match.sentence.words[word_i]
                    obj_counts[obj] += 1
            more_results = False
        except (ConnectionResetError, RequestException) as connection_error:
            query_match = construct_query(spike_engine, spike_annotator, spike_query, continuation_token)

    distinct_objects = len(obj_counts)
    distinct_queries = sum(obj_counts.values())
    print('total number of unique objects: ', distinct_objects)
    print('total number of objects: ', distinct_queries)

    wandb.run.summary['unique_objects'] = distinct_objects
    wandb.run.summary['unique_queries'] = distinct_queries

    dump_json(obj_counts, args.spike_results)


if __name__ == '__main__':
    main()
