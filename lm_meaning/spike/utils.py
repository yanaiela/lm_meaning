import json

from spike.search.data_set_connections import get_data_sets_connections
from pathlib import Path

from spike.search.queries.common.match import SearchMatch
from spike.datamodel.definitions import Sentence
from spike.search.expansion.types import Span
from spike.search.queries.structured.compilation import extract_scaffolding_from_query_text
from spike.exploration import ALGO_DICT


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


def create_match(query: str, annotator) -> SearchMatch:
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


def equal_queries(q1, q2, annotator) -> bool:
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
