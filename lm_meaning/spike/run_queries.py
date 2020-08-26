import argparse
import json
from collections import defaultdict
from pathlib import Path

from spike.search.data_set_connections import get_data_sets_connections
from spike.search.queries.q import StructuredSearchQuery
from tqdm import tqdm


def get_spike_objects(config_path='./my_config.yaml'):
    data_sets_connections = get_data_sets_connections(Path(config_path))
    engine = data_sets_connections.of("wiki").engine
    annotator = data_sets_connections.of("wiki").annotator
    return engine, annotator


def get_relations_data(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()

    lines = [json.loads(x) for x in lines]
    return lines


def get_patterns(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    return lines


def dump_json(data, out_file):
    with open(out_file, 'w') as f:
        json.dump(data, f)


def construct_query(engine, annotator, objs, query_str):
    # aired on relation: P449
    filt_objs = ['`' + x + '`' for x in objs]

    query_with_objs = query_str.format('|'.join(filt_objs))

    search_query = StructuredSearchQuery(query_with_objs, annotator=annotator)
    query_match = engine.match(search_query)
    return query_match


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-data_file", "--data_file", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/TREx_train/P449.jsonl")
    parse.add_argument("-spike_results", "--spike_results", type=str, help="output file to store queries results",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/spike_results/P449.json")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="pattern file",
                       default="/home/lazary/workspace/thesis/lm_meaning/data/spike_patterns/P449.txt")
    # parse.add_argument("-pattern", "--pattern", type=str, help="relation pattern",
    #                    default="")

    args = parse.parse_args()

    spike_engine, spike_annotator = get_spike_objects()

    relations = get_relations_data(args.data_file)
    patterns = get_patterns(args.spike_patterns)

    obj_dic = defaultdict(list)
    for row in relations:
        obj_dic[row['obj_label']].append(row['sub_label'])

    data_dic = defaultdict(dict)
    for row in relations:
        data_dic[row['obj_label']][row['sub_label']] = {}
    for pattern in tqdm(patterns):
        query_match = construct_query(spike_engine, spike_annotator, obj_dic.keys(), pattern)

        for r in tqdm(query_match):
            obj = r.sentence.words[r.captures['object'].first]
            subj = r.sentence.words[r.captures['subject'].first]
            if obj in obj_dic:
                if subj in obj_dic[obj]:
                    sentence = r.sentence.words
                    data_dic[obj][subj][pattern] = sentence

    c = 0
    for obj, v in data_dic.items():
        for subj, vv in v.items():
            c += len(vv)
    print('total number of objects found with all queries', c)

    dump_json(data_dic, args.spike_results)


if __name__ == '__main__':
    main()
