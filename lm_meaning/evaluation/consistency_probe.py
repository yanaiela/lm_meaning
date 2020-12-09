import argparse
from collections import defaultdict
from typing import List, Dict

import numpy as np
from scipy.stats import entropy

import wandb

from lm_meaning.evaluation.paraphrase_comparison import read_json_file
from lm_meaning.spike_patterns.graph_types import EdgeType
from lm_meaning.utils import read_jsonl_file, read_graph


def log_wandb(args):
    pattern = args.trex.split('/')[-1].split('.')[0]
    lm = args.lm_file.split('/')[-1].split('.')[0].split('_')[-1]
    config = dict(
        pattern=pattern,
        lm=lm
    )

    wandb.init(
        name=f'{pattern}_consistency_probe_{lm}',
        project="memorization",
        tags=["eval", pattern, 'paraphrase', lm],
        config=config,
    )


def read_txt_lines(in_f: str) -> List[str]:
    with open(in_f, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def get_first_object(preds, possible_objects):
    for row in preds:
        token = row['token_str']
        if token in possible_objects:
            return token
    return ''


def parse_lm_results(lm_results: Dict, possible_objects: List[str]) -> Dict:
    output_dic = defaultdict(dict)
    c = 0
    for pattern, dic in lm_results.items():
        for data, preds in zip(dic['data'], dic['predictions']):
            subj = data['sub_label']
            obj = data['obj_label']
            key = '_SPLIT_'.join([subj, obj])
            # if key not in output_dic:
            #     output_dic[key] = []
            first_object = get_first_object(preds, possible_objects)
            output_dic[pattern][subj] = first_object
            # if first_object == obj:
            #     output_dic[key].append(pattern)
            # if first_object == '':
            #     c += 1
    # print('number of empties:', c)
    # print('out of:', len(lm_results) * len(list(lm_results.values())[0]['data']))
    return output_dic


def get_node(graph, pattern):
    for node in graph.nodes:
        if node.lm_pattern == pattern:
            return node
    return None


def analyze_results(lm_results: Dict, patterns_graph, subj2obj: Dict) -> None:

    total = 0
    points = 0

    total_syn = 0
    total_lex = 0
    total_both = 0
    total_uni = 0
    total_bi = 0

    points_syn = 0
    points_lex = 0
    points_both = 0
    points_uni = 0
    points_bi = 0

    points_by_edge = defaultdict(list)
    edges_out = defaultdict(list)

    avg_entropy = []

    for pattern, vals in lm_results.items():
        for subj, pred in vals.items():
            graph_node = get_node(patterns_graph, pattern)
            if graph_node is None:
                continue

            base_pattern_success = []
            # going over all entailed patterns
            for ent_node in patterns_graph.successors(graph_node):
                if [graph_node, ent_node] not in patterns_graph.edges:
                    continue
                entailment_type = patterns_graph.edges[graph_node, ent_node]

                ent_pattern = ent_node.lm_pattern
                success = pred == lm_results[ent_pattern][subj]
                if success:
                    points += 1
                total += 1
                base_pattern_success.append(int(success))

                points_by_edge[graph_node.lm_pattern + '_' + ent_node.lm_pattern].append(int(success))
                edges_out[graph_node.lm_pattern].append(int(success))

                if entailment_type['edge_type'] == EdgeType.syntactic:
                    if success:
                        points_syn += 1
                    total_syn += 1
                elif entailment_type['edge_type'] == EdgeType.lexical:
                    if success:
                        points_lex += 1
                    total_lex += 1
                else:
                    if success:
                        points_both += 1
                    total_both += 1

                if [ent_node, graph_node] in patterns_graph.edges:
                    if success:
                        points_bi += 1
                    total_bi += 1
                else:
                    if success:
                        points_uni += 1
                    total_uni += 1

            base_success = sum(base_pattern_success) / len(base_pattern_success)
            ent = entropy([base_success, 1.0 - base_success], base=2)
            avg_entropy.append(ent)

    if total > 0:
        print('overall', points, total, points / total)
        wandb.run.summary['inferred_acc'] = points / total
    else:
        wandb.run.summary['inferred_acc'] = -1
    if total_syn > 0:
        wandb.run.summary['syntactic_inferred_acc'] = points_syn / total_syn
        print('syntactic', points_syn, total_syn, points_syn / total_syn)
    else:
        wandb.run.summary['syntactic_inferred_acc'] = -1
    if total_lex > 0:
        wandb.run.summary['lexical_inferred_acc'] = points_lex / total_lex
        print('lexical', points_lex, total_lex, points_lex / total_lex)
    else:
        wandb.run.summary['lexical_inferred_acc'] = -1
    if total_both > 0:
        print('both', points_both, total_both, points_both / total_both)
        wandb.run.summary['both_inferred_acc'] = points_both / total_both
    else:
        wandb.run.summary['both_inferred_acc'] = -1

    if total_uni > 0:
        print('uni', points_uni, total_uni, points_uni / total_uni)
        wandb.run.summary['uni_inferred_acc'] = points_uni / total_uni
    else:
        wandb.run.summary['uni_inferred_acc'] = -1
    if total_bi > 0:
        print('bi', points_bi, total_bi, points_bi / total_bi)
        wandb.run.summary['bi_inferred_acc'] = points_bi / total_bi
    else:
        wandb.run.summary['bi_inferred_acc'] = -1

    avg_by_edge = []
    for _, vals in points_by_edge.items():
        avg_by_edge.append(sum(vals) / len(vals))

    wandb.run.summary['avg_inferred_by_edge'] = np.average(avg_by_edge)

    avg_out_normalized = []
    out_edges_total = 0
    for k, vals in points_by_edge.items():
        eo = sum(edges_out[k.split('_')[0]]) / len(edges_out[k.split('_')[0]])
        avg_out_normalized.append(eo * (sum(vals) / len(vals)))
        out_edges_total += eo

    wandb.run.summary['avg_inferred_by_edge_out'] = sum(avg_out_normalized) / out_edges_total

    avg_in_normalized = []
    in_edges_total = 0
    for k, vals in points_by_edge.items():
        ei = sum(edges_out[k.split('_')[1]]) / len(edges_out[k.split('_')[1]])
        avg_in_normalized.append(ei * (sum(vals) / len(vals)))
        in_edges_total += ei

    wandb.run.summary['avg_inferred_by_edge_in'] = sum(avg_in_normalized) / in_edges_total

    wandb.run.summary['total'] = total
    wandb.run.summary['total_syn'] = total_syn
    wandb.run.summary['total_lex'] = total_lex
    wandb.run.summary['total_both'] = total_both
    wandb.run.summary['total_bi'] = total_bi
    wandb.run.summary['total_uni'] = total_uni

    wandb.run.summary['avg_entropy'] = np.average(avg_entropy)
    wandb.run.summary['std_entropy'] = np.std(avg_entropy)


def analyze_graph(patterns_graph):
    syn_edges = 0
    lex_edges = 0
    both_edges = 0
    bi_edges = 0
    uni_edges = 0

    for node in patterns_graph:
        for ent_node in patterns_graph.successors(node):
            entailment_type = patterns_graph.edges[node, ent_node]['edge_type']
            if entailment_type == EdgeType.syntactic:
                syn_edges += 1
            elif entailment_type == EdgeType.lexical:
                lex_edges += 1
            else:
                both_edges += 1
            if [ent_node, node] in patterns_graph.edges:
                bi_edges += 1
            else:
                uni_edges += 1

    wandb.run.summary['n_patterns'] = len(patterns_graph)
    wandb.run.summary['all_edges'] = len(patterns_graph.edges)
    wandb.run.summary['syntactic_edges'] = syn_edges
    wandb.run.summary['lexical_edges'] = lex_edges
    wandb.run.summary['both_edges'] = both_edges
    wandb.run.summary['bi_edges'] = bi_edges / 2  # counting these edges twice, so dividing by 2
    wandb.run.summary['uni_edges'] = uni_edges


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-trex", "--trex", type=str, help="trex data file",
                       default="data/trex/data/TREx/P449.jsonl")
    parse.add_argument("-lm_file", "--lm_file", type=str, help="lm prediction file",
                       default="data/predictions_lm/P449_bert-large-cased.json")
    parse.add_argument("-graph", "--graph", type=str, help="graph file",
                       default="data/spike_patterns/graphs/P449.graph")

    args = parse.parse_args()
    log_wandb(args)

    lm_raw_results = read_json_file(args.lm_file)
    patterns_graph = read_graph(args.graph)

    data = read_jsonl_file(args.trex)
    subj_obj = {}
    for row in data:
        subj_obj[row['sub_label']] = row['obj_label']

    all_objects = list(set(subj_obj.values()))

    lm_results = parse_lm_results(lm_raw_results, all_objects)

    analyze_results(lm_results, patterns_graph, subj_obj)
    analyze_graph(patterns_graph)


if __name__ == '__main__':
    main()
