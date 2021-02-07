from glob import glob

import pandas as pd
import spacy
import streamlit as st
from spike.spacywrapper.annotator import SpacyAnnotator

from lm_meaning.spike.utils import equal_queries, lexical_difference
from pararel.consistency.utils import read_jsonl_file, read_graph

st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def get_annotators():
    spike_annotator = SpacyAnnotator.from_config("en.json")
    nlp = spacy.load('en_core_web_sm')
    return spike_annotator, nlp


all_relations = []
for relation in glob('data/pattern_data/graphs/*.graph'):
    all_relations.append(relation.split('/')[-1].split('.')[0])

st.title('Exploration')
# st.write(pattern_id)

relations_file = 'data/trex/data/relations.jsonl'
all_patterns = read_jsonl_file(relations_file)
relations2labels = {x['relation']: x['label'] for x in all_patterns}

relevants_relations = [x for x in all_relations if x in relations2labels.keys()]
relation_names = [f'{relations2labels[x]} ({x})' for x in relevants_relations]


selected_pattern = st.sidebar.radio('Choose Relation', relation_names, index=6)
pattern_id = selected_pattern.split(' ')[-1][1:-1]

# lm = 'bert-base-cased'
graph_file = f'data/pattern_data/graphs/{pattern_id}.graph'

patterns_graph = read_graph(graph_file)


def get_results(patterns_graph, spike_annotator, nlp_annotator):
    connections = []
    for graph_node in patterns_graph.nodes:

        # going over all entailed patterns
        for ent_node in patterns_graph.successors(graph_node):
            # if [graph_node, ent_node] not in patterns_graph.edges:
            #     continue
            syntactic_equal = equal_queries(graph_node.spike_pattern, ent_node.spike_pattern, spike_annotator)
            other_results = lexical_difference(graph_node.lm_pattern, ent_node.lm_pattern, nlp_annotator)

            connections.append(
                [graph_node.lm_pattern, ent_node.lm_pattern, syntactic_equal,
                 not other_results['diff_lemma'], not other_results['diff_det']])
    return connections


spike_annotator, nlp_annotator = get_annotators()

connections = get_results(patterns_graph, spike_annotator, nlp_annotator)

df = pd.DataFrame(connections, columns=['base pattern', 'entailment pattern', 'same-syn', 'same-lemma', 'same-det'])
st.write(df)
