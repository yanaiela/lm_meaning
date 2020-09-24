import argparse
import signal
from collections import defaultdict

from tqdm import tqdm
from typing import List, Iterator

from lm_meaning.spike.utils import get_spike_objects, get_relations_data, dump_json
from spike.annotators.annotator_service import Annotator
from spike.search.engine import MatchEngine
from spike.search.queries.common.match import SearchMatch
from spike.search.queries.q import StructuredSearchQuery

import wandb


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


def construct_query(engine: MatchEngine, annotator: Annotator, objs: List[str], query_str: str
                    ) -> Iterator[SearchMatch]:
    filt_objs = ['`' + x + '`' for x in objs]

    query_with_objs = query_str.format('|'.join(filt_objs))

    search_query = StructuredSearchQuery(query_with_objs, annotator=annotator)
    query_match = engine.match(search_query)
    return query_match


class TimeoutException(Exception):  # custom exception
    pass


def timeout_handler(signum, frame):  # raises exception when signal sent
    raise TimeoutException


# Makes it so that when SIGALRM signal sent, it calls the function timeout_handler, which raises your exception
signal.signal(signal.SIGALRM, timeout_handler)


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

        pbar = tqdm()
        while True:
            signal.alarm(15)
            try:
                r = next(query_match, None)
                if r is None:
                    break
                obj = r.sentence.words[r.captures['object'].first]
                # to capture multi words expressions
                subj = ' '.join(r.sentence.words[r.captures['subject'].first: r.captures['subject'].last + 1])
                if obj in obj_dic:
                    if subj in obj_dic[obj]:
                        sentence = r.sentence.words
                        data_dic[obj][subj][pattern] = sentence
                pbar.update(1)
            except TimeoutException:
                print('time out exception')
                break
        # for r in tqdm(query_match):
        #     obj = r.sentence.words[r.captures['object'].first]
        #     subj = r.sentence.words[r.captures['subject'].first]
        #     if obj in obj_dic:
        #         if subj in obj_dic[obj]:
        #             sentence = r.sentence.words
        #             data_dic[obj][subj][pattern] = sentence

    c = 0
    for obj, v in data_dic.items():
        for subj, vv in v.items():
            c += len(vv)
    print('total number of objects found with all queries', c)

    dump_json(data_dic, args.spike_results)


if __name__ == '__main__':
    main()
