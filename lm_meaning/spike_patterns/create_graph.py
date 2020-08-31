import json
import sys
from typing import List, Tuple


def load_lemmas_relations(fname: str) -> List[Tuple]:

    """
    Return a list of tuples (lemma1, lemma2).
    (lemma1, lemma2) means that lemma1 entails lemma2, e.g. "aired on" entails "broadcasted on".
    """
    with open(fname, "r") as f:
        lines = f.readlines()
    
    lemmas = []
    for line in lines:
    
        left,right = line.strip().split("\t")
        right = right.split(",")
        for r in right:
        
            lemmas.append((left, r))
    
    return lemmas


def load_data(fname: str) -> List[dict]:
    with open(fname, "r") as f:
        lines = f.readlines()
    
    dicts = [eval(l.strip()) for l in lines]
    return dicts
    
    
def get_neighbors(pattern: dict, all_patterns: List[dict], enforce_tense: bool, lemmas_rules: List[Tuple]) -> List[dict]:

    """
    :param pattern: the pattern for which we look for neighbors in the graph (a dictionary)
    :param all_patterns: the list of all patterns
    :parma enforce_tense: if true, look for patterns having the same tense
    :param lemmas_rules: rules describing implicature between lemmas (list of tuples)
    :return: a list of patterns that also hold if this pattern holds.
    """
    
    lemma = pattern["lemma"]
    tense = pattern["tense"]
    relevant_lemmas = set([l2 for (l1,l2) in lemmas_rules if l1 == lemma])
    relevant_patterns = [p for p in all_patterns if p != pattern and p["lemma"] in relevant_lemmas]
    if enforce_tense:
    
        relevant_patterns = [p for p in relevant_patterns if p["tense"] == tense]
    
    return relevant_patterns


if __name__ == "__main__":    
    json_fname = sys.argv[1] #P449.tsv.jsonl
    lemmas_fname = sys.argv[2] #p449_entailment_lemmas.tsv
    patterns = load_data(json_fname)
    lemmas_rules = load_lemmas_relations(lemmas_fname)
    
    pattern = patterns[0]
    neighbors = get_neighbors(pattern, patterns, False, lemmas_rules)
    
    print("Neighbors of the pattern '{}' are:".format(pattern["pattern"]))
    for n in neighbors:
        print(n["pattern"])
