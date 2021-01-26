import argparse
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict
import pandas as pd

import networkx as nx
import tqdm
from spike.spacywrapper.annotator import SpacyAnnotator
import spacy

import wandb

from lm_meaning.spike.utils import equal_queries, lexical_difference
from lm_meaning.spike_patterns.graph_types import PatternNode, EdgeType
from lm_meaning.utils import read_jsonl_file


def log_wandb(args):
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    config = dict(
        pattern=pattern,
    )

    wandb.init(
        entity='consistency',
        name=f'{pattern}_create_graph',
        project="consistency",
        tags=[pattern],
        config=config,
    )


def load_lemmas_relations(fname: str) -> Dict[str, List[str]]:
    """
        Return a dictionary mapping from lemmas to a list of lemmas NOT entailed by it.
        """
    with open(fname, "r") as f:
        lines = f.readlines()

    lemma2not_entailed = defaultdict(list)
    for l in lines[1:]:
        lemma, not_entailed = l.strip().split("\t")
        if not_entailed == '-':
            not_entailed = []
        else:
            not_entailed = not_entailed.split(",")

        lemma2not_entailed[lemma] = not_entailed
    return lemma2not_entailed


def get_neighbors(pattern: dict, all_patterns: List[dict], enforce_tense: bool,
                  lemma2not_entailed: Dict[str, List[str]],
                  connections: dict, spike_annotator,
                  spacy_annotator) -> List[dict]:
    """
    :param lemma2not_entailed:
    :param pattern: the pattern for which we look for neighbors in the graph (a dictionary)
    :param all_patterns: the list of all patterns
    :parma enforce_tense: if true, look for patterns having the same tense
    :param lemmas_rules: rules describing implicature between lemmas (list of tuples)
    :return: a list of patterns that also hold if this pattern holds.
    """

    lemma = pattern["extended_lemma"]
    tense = pattern["tense"]
    relevant_lemmas = set([l for l in list(lemma2not_entailed.keys()) if l not in lemma2not_entailed[lemma]])

    relevant_patterns = [p for p in all_patterns if p != pattern and p["extended_lemma"] in relevant_lemmas]
    if enforce_tense:
        relevant_patterns = [p for p in relevant_patterns if p["tense"] == tense]

    # print("==================================")
    # print("{} neighbors".format(len(relevant_patterns)))
    for r_p in tqdm.tqdm(relevant_patterns, total=len(relevant_patterns)):
        # print(r_p["pattern"], pattern["pattern"])
        syntactic_equivalence = equal_queries(r_p["spike_query"], pattern["spike_query"], spike_annotator)
        lexical_results = lexical_difference(r_p['pattern'], pattern['pattern'], spacy_annotator)
        patterns_diff = {'diff_lemma': lexical_results['diff_lemma'],
                         'diff_det': lexical_results['diff_det'],
                         'diff_syntax': not syntactic_equivalence
                         }
        connections[pattern["pattern"]].append((r_p["pattern"], patterns_diff))

    return relevant_patterns


def filter_dependent_patterns(patterns: List[Dict]) -> List[Dict]:
    """
    Filtering patterns that the lama tuples data (subjects, objects) are mixed, and do not always fit these objects.
    :param patterns: list of all patterns
    :return: filtered list of all patterns, that are not dependent on the subjects
    """
    filtered_patterns = []
    for p in patterns:
        if p['pattern'].startswith('*') or p['pattern'].startswith('#'):
            continue
        filtered_patterns.append(p)
    return filtered_patterns


if __name__ == "__main__":

    parse = argparse.ArgumentParser("")
    parse.add_argument("-patterns_file", "--patterns_file", type=str, help="pattern file",
                       default="data/pattern_data/parsed/P449.jsonl")
    parse.add_argument("-lemmas_file", "--lemmas_file", type=str, help="lemmas file",
                       default="data/pattern_data/entailed_lemmas_extended/P449.tsv")
    parse.add_argument("-out_file", "--out_file", type=str, help="output file",
                       default="data/pattern_data/graphs/P449.graph")
    parse.add_argument("-tense_file", "--tense_file", type=str, help="output file",
                       default="data/pattern_data/memorization_tense.csv")

    args = parse.parse_args()
    log_wandb(args)
    pattern_id = args.patterns_file.split('/')[-1].split('.')[0]
    entailed_from_base = True

    nlp = spacy.load('en_core_web_sm')

    patterns = read_jsonl_file(args.patterns_file)
    lemma2not_entailed = load_lemmas_relations(args.lemmas_file)
    graph = nx.DiGraph()
    pattern2node = dict()

    # collection connections dictionary

    print("Calculating connections between nodes...")
    spike_annotator = SpacyAnnotator.from_config("en.json")

    connections = defaultdict(list)
    for pattern in tqdm.tqdm(patterns, total=len(patterns)):
        get_neighbors(pattern, patterns, False, lemma2not_entailed, connections, spike_annotator, nlp)

    base_pattern = patterns[0]
    # entailed from base
    patterns_entailed_from_base = [t[0] for t in connections[base_pattern['pattern']]]
    entailed_patterns = [x for x in patterns if x['pattern'] in patterns_entailed_from_base and
                         base_pattern['pattern'] in [t[0] for t in connections[x['pattern']]]]

    # entailed from tense
    tenses = pd.read_csv(args.tense_file)
    tense_matters = tenses.set_index('PID').to_dict()['tense_matters']
    if tense_matters[pattern_id] == 'Yes':
        tensed_patterns = [x for x in entailed_patterns if x['tense'] in [base_pattern['tense'], '-', '?']]
    else:
        tensed_patterns = entailed_patterns
    if entailed_from_base:
        subset_patterns = [base_pattern] + tensed_patterns
    else:
        subset_patterns = patterns
    print("Creating graph...")
    for p in tqdm.tqdm(subset_patterns, total=len(patterns)):
        pattern_node = PatternNode(p["pattern"], p["spike_query"],
                                   p["lemma"], p["extended_lemma"], p["tense"], p["example"])
        pattern2node[p["pattern"]] = pattern_node
        graph.add_node(pattern_node)

    # fill the graph with that info
    print("Filling the graph...")
    for i, pattern in enumerate(tqdm.tqdm(subset_patterns, total=len(subset_patterns))):
        print("Pattern {}/{}".format(i, len(subset_patterns)))
        pattern_str = pattern["pattern"]
        connected_patterns, types = zip(*connections[pattern_str])
        for pattern_str2, typ in zip(connected_patterns, types):
            if pattern_str2 not in pattern2node: continue
            node1, node2 = pattern2node[pattern_str], pattern2node[pattern_str2]
            # different_lemma = node1.extended_lemma != node2.extended_lemma
            edge_type = EdgeType(typ['diff_syntax'], typ['diff_lemma'], typ['diff_det'])

            graph.add_edge(node1, node2, edge_type=edge_type)

    with open(args.out_file, "wb") as f:
        pickle.dump(graph, f)
