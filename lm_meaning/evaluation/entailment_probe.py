import argparse
import pickle
from typing import List, Dict

import wandb

from lm_meaning.evaluation.paraphrase_comparison import read_json_file, read_jsonline_file
from lm_meaning.utils import read_jsonl_file


def log_wandb(args):
    pattern = args.lm_patterns.split('/')[-1].split('.')[0]
    lm = args.lm_file.split('/')[-1].split('.')[0].split('_')[-1]
    config = dict(
        pattern=pattern,
        lm=lm
    )

    wandb.init(
        name=f'{pattern}_entailment_probe_{lm}',
        project="memorization",
        tags=["eval", pattern, 'paraphrase', lm],
        config=config,
    )


def read_txt_lines(in_f: str) -> List[str]:
    with open(in_f, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def read_graph(in_file: str):
    with open(in_file, 'rb') as f:
        graph = pickle.load(f)
    return graph


def match_spike_lm_patterns(spike_patterns: List[str], lm_patterns: List[str]) -> Dict:
    spike2lm = {}
    for spike, lm in zip(spike_patterns, lm_patterns):
        spike2lm[spike] = lm
    return spike2lm


def parse_spike_results(spike_results: Dict) -> Dict:
    output_dic = {}
    for obj, dic in spike_results.items():
        for subj, inner_dic in dic.items():
            output_dic['_SPLIT_'.join([subj, obj])] = list(inner_dic.keys())
    return output_dic


def get_first_object(preds, possible_objects):
    for row in preds:
        token = row['token_str']
        if token in possible_objects:
            return token
    return ''


def parse_lm_results(lm_results: Dict, possible_objects: List[str]) -> Dict:
    output_dic = {}
    c = 0
    for pattern, dic in lm_results.items():
        for data, preds in zip(dic['data'], dic['predictions']):
            subj = data['sub_label']
            obj = data['obj_label']
            key = '_SPLIT_'.join([subj, obj])
            if key not in output_dic:
                output_dic[key] = []
            first_object = get_first_object(preds, possible_objects)
            if first_object == obj:
                output_dic[key].append(pattern)
            if first_object == '':
                c += 1
    print('number of empties:', c)
    print('out of:', len(lm_results) * len(list(lm_results.values())[0]['data']))
    return output_dic


def get_node(graph, pattern):
    for node in graph.nodes:
        if node.lm_pattern == pattern:
            return node
    return None


def analyze_results(lm_results: Dict, patterns_graph, spike2lm: Dict, subj2obj: Dict) -> None:

    total = 0
    points = 0

    total_syn = 0
    total_rest = 0
    points_syn = 0
    points_rest = 0

    # all_patterns = spike2lm.values()

    for key, vals in lm_results.items():
        subj, obj = key.split('_SPLIT_')
        for successful_lm_pattern in vals:
            graph_node = get_node(patterns_graph, successful_lm_pattern)
            if graph_node is None:
                # print(successful_lm_pattern)
                continue

            # going over all entailed patterns
            for ent_node in patterns_graph.successors(graph_node):
                if [ent_node, graph_node] not in patterns_graph.edges:
                    continue
                entailment_type = patterns_graph.edges[ent_node, graph_node]
                if len(entailment_type) == 1:
                    entailment_type = 'syn'
                else:
                    entailment_type = 'rest'

                ent_pattern = ent_node.lm_pattern
                # going over all data
                for new_subj, new_obj in subj2obj.items():
                    # in case these are the same
                    if new_subj == subj and new_obj == obj:
                        continue
                    new_key = '{}_SPLIT_{}'.format(new_subj, new_obj)
                    if ent_pattern in lm_results[new_key]:
                        points += 1
                    total += 1

                    if entailment_type == 'syn':
                        if ent_pattern in lm_results[new_key]:
                            points_syn += 1
                        total_syn += 1
                    else:
                        if ent_pattern in lm_results[new_key]:
                            points_rest += 1
                        total_rest += 1

    print(points, total, points / total)
    print(points_syn, total_syn, points_syn / total_syn)
    print(points_rest, total_rest, points_rest / total_rest)
    wandb.run.summary['inferred_acc'] = points / total
    wandb.run.summary['syntactic_inferred_acc'] = points_syn / total_syn
    wandb.run.summary['lexical_inferred_acc'] = points_rest / total_rest


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-trex", "--trex", type=str, help="trex data file",
                       default="data/trex/data/TREx/P449.jsonl")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file",
                       default="data/predictions_lm/P449_bert-large-cased.json")
    parse.add_argument("-lm_patterns", "--lm_patterns", type=str, help="lm patterns",
                       default="data/lm_relations/P449.jsonl")
    parse.add_argument("-spike_patterns", "--spike_patterns", type=str, help="spike pattern",
                       default="data/spike_patterns/P449.txt")
    parse.add_argument("-graph", "--graph", type=str, help="graph file",
                       default="data/spike_patterns/graphs/P449.graph")

    args = parse.parse_args()
    log_wandb(args)

    lm_patterns = [x['pattern'] for x in read_jsonline_file(args.spike_patterns)]
    spike_patterns = [x['spike_query'] for x in read_jsonline_file(args.spike_patterns)]
    spike2lm = match_spike_lm_patterns(spike_patterns, lm_patterns)

    lm_raw_results = read_json_file(args.lm_file)
    patterns_graph = read_graph(args.graph)

    data = read_jsonl_file(args.trex)
    subj_obj = {}
    for row in data:
        subj_obj[row['sub_label']] = row['obj_label']

    all_objects = list(set(subj_obj.values()))

    lm_results = parse_lm_results(lm_raw_results, all_objects)

    analyze_results(lm_results, patterns_graph, spike2lm, subj_obj)


if __name__ == '__main__':
    main()
