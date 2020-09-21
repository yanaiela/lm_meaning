import argparse
from collections import defaultdict
from typing import Iterator, Optional

from spike.annotators.annotator_service import Annotator
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.search.queries.q import StructuredSearchQuery
from spike.integration.odinson.common import OdinsonContinuationToken

from tqdm import tqdm
import wandb
from lm_meaning.spike.utils import get_spike_objects, get_relations_data, dump_json


def log_wandb(args):
    pattern = args.spike_patterns.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
    )

    wandb.init(
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


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/pattern_data/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/output/spike_results/pattern_counts"
                               "/P449.json")

    args = parse.parse_args()
    log_wandb(args)

    spike_engine, spike_annotator = get_spike_objects()

    patterns = [x['spike_query'] for x in get_relations_data(args.spike_patterns)]

    data_dic = defaultdict(int)
    # The following code is slightly complicated due to the continuation token option, which allows to reset the
    #  iterator which is connected to spike, to avoid connection issues.
    # The code simply reset every 500K results, and then uses the continuation token in order to continue from the
    #  point where it stopped.
    for pattern in tqdm(patterns):
        more_results = True
        count = 0
        continuation_token = None
        while more_results:
            query_match = construct_query(spike_engine, spike_annotator, pattern.replace('[w={}]', ''),
                                          continuation=continuation_token)
            for r in tqdm(query_match):
                if r is None:
                    more_results = False
                    break
                count += 1
                if count % 500000 == 0:
                    continuation_token = r.continuation_token
                    if continuation_token is None:
                        more_results = False
                        break
            if next(query_match, None) is None:
                more_results = False

        data_dic[pattern] = count

    for pattern, count in data_dic.items():
        print('pattern: {}. count: {}'.format(pattern, count))

    table = wandb.Table(columns=["Pattern", "Count"])
    for pattern, count in data_dic.items():
        table.add_data(pattern, count)
    wandb.log({"patterns_results": table})

    dump_json(data_dic, args.spike_results)


if __name__ == '__main__':
    main()
