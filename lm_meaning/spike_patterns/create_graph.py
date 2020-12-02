import argparse
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict

import networkx as nx
import tqdm
from spike.spacywrapper.annotator import SpacyAnnotator

import wandb

from lm_meaning.spike.utils import equal_queries
from lm_meaning.spike_patterns.graph_types import PatternNode, EdgeType
from lm_meaning.utils import read_jsonl_file


def log_wandb(args):
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    config = dict(
        pattern=pattern,
    )

    wandb.init(
        name=f'{pattern}_create_graph',
        project="memorization",
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


def get_neighbors(pattern: dict, all_patterns: List[dict], enforce_tense: bool, lemma2not_entailed: List[Tuple],
                  connections: dict, spike_annotator) -> List[dict]:
    """
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
        if equal_queries(r_p["spike_query"], pattern["spike_query"], spike_annotator):
            syntactic_equivalence = True
        else:
            syntactic_equivalence = False
        connections[pattern["pattern"]].append((r_p["pattern"],
                                                "same_syntax" if syntactic_equivalence else "diff_syntax"))

    return relevant_patterns


if __name__ == "__main__":

    parse = argparse.ArgumentParser("")
    parse.add_argument("-patterns_file", "--patterns_file", type=str, help="pattern file",
                       default="data/output/P449.jsonl")
    parse.add_argument("-lemmas_file", "--lemmas_file", type=str, help="lemmas file",
                       default="data/output/P449_entailment_lemmas.tsv")
    parse.add_argument("-out_file", "--out_file", type=str, help="output file",
                       default="data/pattern_data/graphs/P449.graph")

    args = parse.parse_args()
    log_wandb(args)

    patterns = read_jsonl_file(args.patterns_file)
    lemma2not_entailed = load_lemmas_relations(args.lemmas_file)
    graph = nx.DiGraph()
    pattern2node = dict()

    print("Creating graph...")
    for p in tqdm.tqdm(patterns, total=len(patterns)):
        pattern_node = PatternNode(p["pattern"], p["spike_query"],
                                   p["lemma"], p["extended_lemma"], p["tense"], p["example"])
        pattern2node[p["pattern"]] = pattern_node
        graph.add_node(pattern_node)

    # collection connections dictionary

    print("Calcualting connections between nodes...")
    spike_annotator = SpacyAnnotator.from_config("en.json")

    connections = defaultdict(list)
    for pattern in tqdm.tqdm(patterns, total=len(patterns)):
        get_neighbors(pattern, patterns, False, lemma2not_entailed, connections, spike_annotator)

    # fill the graph with that info
    print("Filling the graph...")
    for i, pattern in enumerate(tqdm.tqdm(patterns, total=len(patterns))):
        print("Pattern {}/{}".format(i, len(patterns)))
        pattern_str = pattern["pattern"]
        connected_patterns, types = zip(*connections[pattern_str])
        for pattern_str2, typ in zip(connected_patterns, types):
            node1, node2 = pattern2node[pattern_str], pattern2node[pattern_str2]
            different_lemma = node1.extended_lemma != node2.extended_lemma
            if typ == "diff_syntax" and different_lemma:  # different lemma, different syntax
                edge_type = EdgeType.both
            elif typ == "diff_syntax":  # different syntax, same lemma
                edge_type = EdgeType.syntactic
            else:  # same syntax, different lemma
                edge_type = EdgeType.lexical

            graph.add_edge(node1, node2, edge_type=edge_type)

    with open(args.out_file, "wb") as f:
        pickle.dump(graph, f)
