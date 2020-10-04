import json
import sys
from typing import List, Tuple, Dict
from collections import defaultdict
from lm_meaning.spike.utils import equal_queries, get_spike_objects
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
from lm_meaning.spike_patterns.graph_types import PatternNode, EdgeType
import random
import matplotlib.pyplot as plt
import tqdm
import pickle

def load_lemmas_relations(fname: str) -> Dict[str, List[str]]:

        """
        Return a dictionary mapping from lemmas to a list of lemmas NOT entailed by it.
        """
        with open(fname, "r") as f:
            lines = f.readlines()
            
        lemma2not_entailed = defaultdict(list)
        for l in lines[1:]:
        
            lemma, not_entailed = l.strip().split("\t")
            not_entailed = not_entailed.split(",")

            
            lemma2not_entailed[lemma] = not_entailed
        return lemma2not_entailed
        
        
        
def load_data(fname: str) -> List[dict]:
    with open(fname, "r") as f:
        lines = f.readlines()
    
    dicts = [eval(l.strip()) for l in lines]
    return dicts
    
    
def get_neighbors(pattern: dict, all_patterns: List[dict], enforce_tense: bool, lemma2not_entailed: List[Tuple], connections: dict, spike_annotator) -> List[dict]:

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
    
    
    #print("==================================")
    #print("{} neighbors".format(len(relevant_patterns)))
    for r_p in tqdm.tqdm(relevant_patterns, total = len(relevant_patterns)):
        #print(r_p["pattern"], pattern["pattern"])
        if equal_queries(r_p["spike_query"], pattern["spike_query"], spike_annotator):
            is_syntactic = True
        else:
            is_syntactic = False
        connections[pattern["pattern"]].append((r_p["pattern"], "syntactic" if is_syntactic else "lexical"))

    return relevant_patterns


if __name__ == "__main__":    
    json_fname = sys.argv[1] #P449.tsv.jsonl
    lemmas_fname = sys.argv[2] #p449_entailment_lemmas.tsv
    patterns = load_data(json_fname)
    lemma2not_entailed = load_lemmas_relations(lemmas_fname)
    graph = nx.DiGraph()
    pattern2node = dict()

    print("Creating graph...")
    for p in tqdm.tqdm(patterns, total = len(patterns)):
        pattern_node = PatternNode(p["pattern"], p["spike_query"],
                               p["lemma"], p["extended_lemma"], p["tense"], p["example"])
        pattern2node[p["pattern"]] = pattern_node
        graph.add_node(pattern_node)

    # collection connections dictionary

    print("Calcualting connections between nodes...")
    _, spike_annotator = get_spike_objects()
    connections = defaultdict(list)
    for pattern in tqdm.tqdm(patterns, total = len(patterns)):

        get_neighbors(pattern, patterns, False, lemma2not_entailed, connections, spike_annotator)

    # fill the graph with that info
    print("Filling the graph...")
    for i, pattern in enumerate(tqdm.tqdm(patterns, total = len(patterns))):
        print("Pattern {}/{}".format(i, len(patterns)))
        pattern_str = pattern["pattern"]
        connected_patterns, types = zip(*connections[pattern_str])
        for pattern_str2, typ in zip(connected_patterns, types):
            node1, node2 = pattern2node[pattern_str], pattern2node[pattern_str2]
            if typ == "syntactic":
                graph.add_edge(node1, node2, edge_type=EdgeType.syntactic)
            else:
                graph.add_edge(node1, node2)

    with open("graphs/" + json_fname.split(".")[0]+"graph", "wb") as f:
        pickle.dump(graph, f)
    nx.draw(graph)
    plt.show()
