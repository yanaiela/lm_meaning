import json
from pathlib import Path
from typing import List, Dict

from functools import lru_cache

from spike.annotators.annotator_service import Annotator
from spike.datamodel.definitions import Sentence
from spike.exploration import ALGO_DICT
from spike.search.data_set_connections import get_data_sets_connections
from spike.search.engine import MatchEngine
from spike.search.expansion.types import Span
from spike.search.queries.common.match import SearchMatch
# from spike.search.queries.structured.compilation import extract_scaffolding_from_query_text


def get_spike_objects(config_path: str = './my_config.yaml') -> (MatchEngine, Annotator):
    data_sets_connections = get_data_sets_connections(Path(config_path))
    engine = data_sets_connections.of("wiki").engine
    annotator = data_sets_connections.of("wiki").annotator
    return engine, annotator


def get_relations_data(in_file: str) -> List[Dict]:
    with open(in_file, 'r') as f:
        lines = f.readlines()

    lines = [json.loads(x) for x in lines]
    return lines


def get_patterns(in_file: str) -> List[str]:
    with open(in_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    return lines


def dump_json(data: Dict, out_file: str):
    with open(out_file, 'w') as f:
        json.dump(data, f)


def create_match(query: str, annotator: Annotator) -> SearchMatch:
    t = extract_scaffolding_from_query_text(query, annotator)
    graph = t.as_graph_representation()
    nodes = graph.nodes
    sentence = Sentence(graph.original_words, [], [], [], [], [], [],
                        {'universal-enhanced': graph.graph}, {}, [])

    subject_ind = [ind for ind, x in enumerate(nodes) if 'subject' in x][0]
    object_ind = [ind for ind, x in enumerate(nodes) if 'object' in x][0]
    subject_span = Span(subject_ind, subject_ind)
    object_span = Span(object_ind, object_ind)

    captures = {'subject': subject_span, 'object': object_span}
    search_match = SearchMatch(sentence, captures, None, None)
    return search_match


def equal_queries(q1: str, q2: str, annotator: Annotator) -> bool:
    q1_clean = q1.replace('[w={}]', '')
    q2_clean = q2.replace('[w={}]', '')
    m1 = create_match(q1_clean, annotator)
    m2 = create_match(q2_clean, annotator)

    graph_algo = ALGO_DICT['group_by_syntax_any_token']
    p1 = graph_algo.get_patterns(m1)
    p2 = graph_algo.get_patterns(m2)

    assert len(p1) == 1
    assert len(p2) == 1
    return list(p1)[0].signature == list(p2)[0].signature


def enclose_entities(annotator: Annotator, entity: str) -> str:
    annotated = annotator.annotate_text(entity)
    words = []
    for sentence in annotated.sentences:
        words.extend(sentence.words)
    return ' '.join(words)


def _lexical_diff(words2pos1, words2pos2):
    words1 = [x[0] for x in words2pos1]
    words2 = [x[0] for x in words2pos2]
    prep_substitute = False
    for lemma, pos in words2pos1:
        if lemma not in words2:
            if pos in ['DET', 'PUNCT', 'SYM']:
                continue
            if pos in ['ADP']:
                prep_substitute = True
                continue
            return True, prep_substitute
        if words1.count(lemma) != words2.count(lemma):
            return True, prep_substitute
    return False, prep_substitute


def _det_diff(words2pos1, words2pos2):
    words1 = [x[0] for x in words2pos1]
    words2 = [x[0] for x in words2pos2]
    for lemma, pos in words2pos1:
        if lemma not in words2:
            if pos in ['DET']:
                return True
        if pos in ['DET'] and words1.count(lemma) != words2.count(lemma):
            return True
    return False


@lru_cache(maxsize=None)
def spacy_annotation(spacy_obj, text):
    return spacy_obj(text)


def lexical_difference(q1, q2, spacy_annotator):
    doc1 = spacy_annotation(spacy_annotator, q1.replace('[X]', 'subject').replace('[Y]', 'object'))
    doc2 = spacy_annotation(spacy_annotator, q2.replace('[X]', 'subject').replace('[Y]', 'object'))
    words1 = [(x.lemma_, x.pos_) for x in doc1 if x.text not in ['subject', 'object']]
    words2 = [(x.lemma_, x.pos_) for x in doc2 if x.text not in ['subject', 'object']]

    diff_lemma, prep_substitutue1 = _lexical_diff(words1, words2)
    if not diff_lemma:
        diff_lemma, prep_substitutue2 = _lexical_diff(words2, words1)
        # if both preposition were substituted, considering it as a lexical change
        # e.g. "[X] died in [Y]." and "[X] died at [Y]."
        if prep_substitutue1 and prep_substitutue2:
            diff_lemma = True

    diff_det = _det_diff(words1, words2)
    if not diff_det:
        diff_det = _det_diff(words2, words1)

    return {'diff_lemma': diff_lemma,
            'diff_det': diff_det}
