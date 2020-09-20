import json
import sys
from typing import List, Tuple, Dict
from collections import defaultdict


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
    
    
def get_neighbors(pattern: dict, all_patterns: List[dict], enforce_tense: bool, lemma2not_entailed: List[Tuple]) -> List[dict]:

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
    
    return relevant_patterns


if __name__ == "__main__":    
    json_fname = sys.argv[1] #P449.tsv.jsonl
    lemmas_fname = sys.argv[2] #p449_entailment_lemmas.tsv
    patterns = load_data(json_fname)
    lemma2not_entailed = load_lemmas_relations(lemmas_fname)
    pattern = patterns[5]
    
    neighbors = get_neighbors(pattern, patterns, False, lemma2not_entailed)
    
    print("Neighbors of the pattern '{}' are:".format(pattern["pattern"]))
    for n in neighbors:
        print(n["pattern"])
