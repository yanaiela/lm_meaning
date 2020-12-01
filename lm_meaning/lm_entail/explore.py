from glob import glob

import pandas as pd
import streamlit as st

from lm_meaning.evaluation.entailment_probe import get_node, parse_lm_results
from lm_meaning.utils import read_json_file, read_jsonl_file, read_graph


def highlight_errors(row):
    return ['background-color: #d65f5f' if not row['success'] else '' for _ in row]


@st.cache()
def get_data(data_file):
    return read_jsonl_file(data_file)


all_relations = []
for relation in glob('data/pattern_data/graphs/*.graph'):
    all_relations.append(relation.split('/')[-1].split('.')[0])

st.title('Exploration')
# st.write(pattern_id)

pattern_id = st.sidebar.radio('Relation', all_relations)

lm = 'bert-base-cased'
data_file = f'data/trex_lms_vocab/{pattern_id}.jsonl'
lm_file = f'data/output/predictions_lm/trex_lms_vocab/{pattern_id}_{lm}.json'
graph_file = f'data/pattern_data/graphs/{pattern_id}.graph'

lm_raw_results = read_json_file(lm_file)
patterns_graph = read_graph(graph_file)

data = get_data(data_file)
subj_obj = {}
for row in data:
    subj_obj[row['sub_label']] = row['obj_label']

all_objects = list(set(subj_obj.values()))

lm_results = parse_lm_results(lm_raw_results, all_objects)


def get_results(lm_results, patterns_graph):
    successes = []
    failures = []
    for key, vals in lm_results.items():
        for successful_lm_pattern in vals:
            graph_node = get_node(patterns_graph, successful_lm_pattern)
            if graph_node is None:
                continue

            # going over all entailed patterns
            for ent_node in patterns_graph.successors(graph_node):
                if [graph_node, ent_node] not in patterns_graph.edges:
                    continue
                entailment_type = patterns_graph.edges[graph_node, ent_node]['edge_type'].name

                ent_pattern = ent_node.lm_pattern
                success = ent_pattern in lm_results[key]
                failures.append(
                    [graph_node.lm_pattern, ent_node.lm_pattern, ] + key.split('_SPLIT_') + [entailment_type, success])
    return successes, failures


success_results, failure_results = get_results(lm_results, patterns_graph)

df = pd.DataFrame(failure_results, columns=['base pattern', 'entailment pattern', 'subj', 'obj', 'ent-type', 'success'])
df_style = df.style.apply(highlight_errors, axis=1)
st.write(df_style)

n = len(df)
success_count = df['success'].value_counts()[1]

st.write('overall tuples:', n)
st.write('overall model success:', success_count, success_count / n)
