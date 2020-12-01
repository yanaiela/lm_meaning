from glob import glob

import pandas as pd
import streamlit as st
from tqdm import tqdm

from lm_meaning.evaluation.entailment_probe import get_node, parse_lm_results
from lm_meaning.utils import read_json_file, read_jsonl_file, read_graph


def highlight_errors(row):
    if row['success']:
        return ['background-color: #5fba7d'] * len(row)
    else:
        return ['background-color: #d65f5f'] * len(row)
    # return ['background-color: #d65f5f' if not row['success'] else 'background-color: #5fba7d' for _ in row]


@st.cache()
def get_data(data_file):
    return read_jsonl_file(data_file)


all_relations = []
for relation in glob('data/pattern_data/graphs/*.graph'):
    all_relations.append(relation.split('/')[-1].split('.')[0])


st.set_page_config(layout="wide")


st.title('Exploration')
# st.write(pattern_id)

possible_lms = ['bert-base-cased', 'bert-large-cased', 'bert-large-cased-whole-word-masking',
                'robert-base', 'roberta-large',
                'albert-base-v2', 'albert-xxlarge-v2']
relations_file = 'data/trex/data/relations.jsonl'
all_patterns = read_jsonl_file(relations_file)
relations2labels = {x['relation']: x['label'] for x in all_patterns}

relevants_relations = [x for x in all_relations if x in relations2labels.keys()]
relation_names = [f'{relations2labels[x]} ({x})' for x in relevants_relations]


count = st.sidebar.slider('N. Results', min_value=10, max_value=1000, value=100)
lm = st.sidebar.radio('LM', possible_lms, index=1)
# pattern_id = st.sidebar.radio('Relation', all_relations)
selected_pattern = st.sidebar.radio('Choose Relation', relation_names, index=6)
pattern_id = selected_pattern.split(' ')[-1][1:-1]

# lm = 'bert-base-cased'
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
    for key, vals in tqdm(lm_results.items()):
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
df = df[:count]
df_style = df.style.apply(highlight_errors, axis=1)
st.write(df_style)

n = len(df)
success_count = df['success'].value_counts()[1]

st.write('overall tuples:', n)
st.write('overall success:', success_count)
st.write('% success:', success_count / n)
