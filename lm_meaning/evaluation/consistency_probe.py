import argparse
from collections import defaultdict
from typing import List, Dict

import numpy as np
import wandb
from scipy.stats import entropy

from lm_meaning.evaluation.paraphrase_comparison import read_json_file
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
            first_object = get_first_object(preds, possible_objects)
            output_dic[pattern][subj] = (first_object, obj)
    return output_dic


def get_node(graph, pattern):
    for node in graph.nodes:
        if node.lm_pattern == pattern:
            return node
    return None


def analyze_results(lm_results: Dict, patterns_graph) -> None:
    total = 0
    points = 0

    total_syn = 0
    total_lex = 0
    total_both = 0
    total_no = 0

    points_syn = 0
    points_lex = 0
    points_both = 0
    points_no = 0

    points_by_edge = defaultdict(list)
    edges_out = defaultdict(list)

    avg_entropy = []

    consistent_subjects = defaultdict(list)
    correct_subjects_per_pattern = defaultdict(int)
    correct_patterns_per_subject = defaultdict(int)
    consistency_performance = defaultdict(list)

    for pattern, vals in lm_results.items():
        for subj, (pred, gold_obj) in vals.items():
            graph_node = get_node(patterns_graph, pattern)
            if graph_node is None:
                continue

            correct_patterns_per_subject[subj] += int(pred == gold_obj)
            correct_subjects_per_pattern[pattern] += int(pred == gold_obj)
            consistent_subjects[subj].append(pred)
            base_pattern_success = []
            # going over all entailed patterns
            for ent_node in patterns_graph.successors(graph_node):
                if [graph_node, ent_node] not in patterns_graph.edges:
                    continue
                entailment_type = patterns_graph.edges[graph_node, ent_node]

                ent_pattern = ent_node.lm_pattern
                success = pred == lm_results[ent_pattern][subj][0]
                if success:
                    points += 1
                total += 1
                base_pattern_success.append(int(success))
                consistency_performance[subj].append(success)

                points_by_edge[graph_node.lm_pattern + '_' + ent_node.lm_pattern].append(int(success))
                edges_out[graph_node.lm_pattern].append(int(success))

                if entailment_type['edge_type'].syntactic_change and not entailment_type['edge_type'].lexical_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_syn += 1
                    total_syn += 1
                elif entailment_type['edge_type'].lexical_change and not entailment_type['edge_type'].syntactic_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_lex += 1
                    total_lex += 1
                elif entailment_type['edge_type'].lexical_change and entailment_type['edge_type'].syntactic_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_both += 1
                    total_both += 1
                if not entailment_type['edge_type'].syntactic_change and not entailment_type['edge_type'].lexical_change \
                        and not entailment_type['edge_type'].determiner_change:
                    if success:
                        points_no += 1
                    total_no += 1

            base_success = sum(base_pattern_success) / len(base_pattern_success)
            ent = entropy([base_success, 1.0 - base_success], base=2)
            avg_entropy.append(ent)

    if total > 0:
        print('overall', points, total, points / total)
        wandb.run.summary['consistency'] = points / total
    else:
        wandb.run.summary['consistency'] = -1
    if total_syn > 0:
        wandb.run.summary['syntactic_consistency'] = points_syn / total_syn
        print('syntactic', points_syn, total_syn, points_syn / total_syn)
    else:
        wandb.run.summary['syntactic_consistency'] = -1
    if total_lex > 0:
        wandb.run.summary['lexical_consistency'] = points_lex / total_lex
        print('lexical', points_lex, total_lex, points_lex / total_lex)
    else:
        wandb.run.summary['lexical_consistency'] = -1
    if total_no > 0:
        wandb.run.summary['no_change_consistency'] = points_no / total_no
        print('no change', points_no, total_no, points_no / total_no)
    else:
        wandb.run.summary['no_change_consistency'] = -1
    if total_both > 0:
        print('both', points_both, total_both, points_both / total_both)
        wandb.run.summary['both_consistency'] = points_both / total_both
    else:
        wandb.run.summary['both_consistency'] = -1

    avg_out_normalized = []
    out_edges_total = 0
    for k, vals in points_by_edge.items():
        eo = sum(edges_out[k.split('_')[0]]) / len(edges_out[k.split('_')[0]])
        avg_out_normalized.append(eo * (sum(vals) / len(vals)))
        out_edges_total += eo
    wandb.run.summary['avg_consistency_by_edge_out'] = sum(avg_out_normalized) / out_edges_total

    all_consistent = 0
    for subj, preds in consistent_subjects.items():
        preds_set = set(preds)
        if len(preds_set) == 1:
            all_consistent += 1
    wandb.run.summary['consistent_subjects'] = all_consistent / len(consistent_subjects)

    successful_subjects = 0
    for subj, success in correct_patterns_per_subject.items():
        if success > 0:
            successful_subjects += 1
    wandb.run.summary['successful_subjects'] = successful_subjects / len(correct_patterns_per_subject)

    successful_patterns = 0
    for pattern, success in correct_subjects_per_pattern.items():
        if success > 0:
            successful_patterns += 1
    wandb.run.summary['successful_patterns'] = successful_patterns / len(correct_subjects_per_pattern)

    success_for_knowledgable_patterns, total_for_knowledgable_patterns = 0, 0
    success_for_unknowledgable_patterns, total_for_unknowledgable_patterns = 0, 0
    for subj, success in consistency_performance.items():
        if correct_patterns_per_subject[subj] > 0:
            success_for_knowledgable_patterns += sum(success)
            total_for_knowledgable_patterns += len(success)
        else:
            success_for_unknowledgable_patterns += sum(success)
            total_for_unknowledgable_patterns += len(success)
    wandb.run.summary['knowledgable_consistency'] = success_for_knowledgable_patterns / total_for_knowledgable_patterns
    wandb.run.summary['unknowledgable_consistency'] = success_for_unknowledgable_patterns \
                                                      / total_for_unknowledgable_patterns

    wandb.run.summary['total'] = total
    wandb.run.summary['total_syn'] = total_syn
    wandb.run.summary['total_lex'] = total_lex
    wandb.run.summary['total_both'] = total_both
    wandb.run.summary['total_no'] = total_no

    wandb.run.summary['avg_entropy'] = np.average(avg_entropy)
    wandb.run.summary['std_entropy'] = np.std(avg_entropy)


def analyze_graph(patterns_graph):
    syn_edges = 0
    lex_edges = 0
    both_edges = 0

    for node in patterns_graph:
        for ent_node in patterns_graph.successors(node):
            entailment_type = patterns_graph.edges[node, ent_node]['edge_type']
            if entailment_type.syntactic_change and not entailment_type.lexical_change \
                    and not entailment_type.determiner_change:
                syn_edges += 1
            elif entailment_type.lexical_change and not entailment_type.syntactic_change \
                    and not entailment_type.determiner_change:
                lex_edges += 1
            elif entailment_type.lexical_change and entailment_type.syntactic_change \
                    and not entailment_type.determiner_change:
                both_edges += 1

    wandb.run.summary['n_patterns'] = len(patterns_graph)
    wandb.run.summary['all_edges'] = len(patterns_graph.edges)
    wandb.run.summary['syntactic_edges'] = syn_edges
    wandb.run.summary['lexical_edges'] = lex_edges
    wandb.run.summary['both_edges'] = both_edges


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

    analyze_results(lm_results, patterns_graph)
    analyze_graph(patterns_graph)


if __name__ == '__main__':
    main()
