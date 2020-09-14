import argparse
from collections import defaultdict

from spike.search.queries.q import StructuredSearchQuery
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


def construct_query(engine, annotator, spike_query):
    search_query = StructuredSearchQuery(spike_query, annotator=annotator)
    query_match = engine.match(search_query)
    return query_match


def construct_spike_query(pattern: str) -> str:
    spike_pattern = pattern.replace('[X]', '').strip()
    tokens = spike_pattern.split()
    spike_tokens = []
    for i in range(len(tokens)):
        if tokens[i] == '[Y]':
            break

        # kinda hacky, but this deals with P449 (the "aired on" relation), which for some reason added
        #  this word, which makes the results much more sparse
        if tokens[i] == 'originally':
            continue
        spike_tokens.append(f'${tokens[i]}')
    spike_tokens.append('object:Object')
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
    wandb.run.summary['spike_query'] = spike_query

    spike_engine, spike_annotator = get_spike_objects()

    obj_counts = defaultdict(int)

    query_match = construct_query(spike_engine, spike_annotator, spike_query)

    for match in tqdm(query_match):
        if match is None:
            break
        obj = match.sentence.words[match.captures['object'].first]
        obj_counts[obj] += 1

    distinct_objects = len(obj_counts)
    distinct_queries = sum(obj_counts.values())
    print('total number of unique objects: ', distinct_objects)
    print('total number of objects: ', distinct_queries)

    wandb.run.summary['unique_objects'] = distinct_objects
    wandb.run.summary['unique_queries'] = distinct_queries

    dump_json(obj_counts, args.spike_results)


if __name__ == '__main__':
    main()
