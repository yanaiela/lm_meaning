import argparse
import signal
from collections import defaultdict
from typing import List, Iterator, Optional

import wandb
from requests.exceptions import RequestException
from spike.annotators.annotator_service import Annotator
from spike.integration.odinson.common import OdinsonContinuationToken
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.search.queries.q import StructuredSearchQuery
from tqdm import tqdm

from lm_meaning.spike.utils import get_spike_objects, get_relations_data, dump_json


def log_wandb(args):
    pattern = args.spike_patterns.split('/')[-1].split('.')[0]

    config = dict(
        pattern=pattern,
    )

    wandb.init(
        name=f'{pattern}_paraphrase_queries',
        project="memorization",
        tags=["spike", pattern, 'paraphrases'],
        config=config,
    )


def construct_query(engine: MatchEngine, annotator: Annotator, objs: List[str], query_str: str,
                    continuation_token: Optional[OdinsonContinuationToken] = None) -> Iterator[SearchMatch]:
    filt_objs = ['`' + x + '`' for x in objs]

    query_with_objs = query_str.format('|'.join(filt_objs))

    search_query = StructuredSearchQuery(query_with_objs, annotator=annotator)
    query_match = engine.match(search_query, continuation=continuation_token)
    return query_match


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-data_file", "--data_file", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/TREx_train/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/spike_results/P449.json")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/spike_patterns/P449.txt")

    args = parse.parse_args()
    log_wandb(args)

    spike_engine, spike_annotator = get_spike_objects()

    relations = get_relations_data(args.data_file)
    patterns = [x['spike_query'] for x in get_relations_data(args.spike_patterns)]

    obj_dic = defaultdict(list)
    for row in relations:
        obj_dic[row['obj_label']].append(row['sub_label'])

    data_dic = defaultdict(dict)
    for row in relations:
        data_dic[row['obj_label']][row['sub_label']] = {}
    for pattern in tqdm(patterns):
        query_match = construct_query(spike_engine, spike_annotator, list(obj_dic.keys()), pattern)

        more_results = True
        continuation_token = None
        while more_results:
            try:
                for r in tqdm(query_match):
                    if r is None:
                        break
                    continuation_token = r.continuation_token
                    obj = r.sentence.words[r.captures['object'].first]
                    # to capture multi words expressions
                    subj = ' '.join(r.sentence.words[r.captures['subject'].first: r.captures['subject'].last + 1])
                    if obj in obj_dic:
                        if subj in obj_dic[obj]:
                            sentence = r.sentence.words
                            data_dic[obj][subj][pattern] = sentence
                more_results = False
            except (ConnectionResetError, RequestException) as connection_error:
                query_match = construct_query(spike_engine, spike_annotator, list(obj_dic.keys()), pattern,
                                              continuation_token)

    c = 0
    for obj, v in data_dic.items():
        for subj, vv in v.items():
            c += len(vv)
    print('total number of objects found with all queries', c)
    wandb.run.summary['n_matches'] = c
    wandb.run.summary['n_patterns'] = len(patterns)
    wandb.run.summary['n_relations'] = len(relations)
    wandb.run.summary['n_subjects'] = len(list(set(obj_dic.values())))
    wandb.run.summary['n_objects'] = len(list(set(obj_dic.keys())))

    dump_json(data_dic, args.spike_results)


if __name__ == '__main__':
    main()
