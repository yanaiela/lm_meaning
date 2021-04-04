from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import StopException

from lm_meaning.explanation.explain import explain_preference_bias, explain_cooccurrences, explain_memorization, \
    explain_subject_contains_object, get_lm_preds, get_items
from pararel.consistency.utils import read_json_file, read_jsonl_file


def highlight_errors(row):
    return ['background-color: red' if (row['memorization'] == -1 and row['cooccurrences'] == -1 and
                                        row['preference'] == -1 and row['contains'] == -1) else '' for _ in row]


bias_file = 'data/preference_bias/bias.json'
relations_file = 'data/trex/data/relations.jsonl'

memorization_dir = 'data/output/spike_results/paraphrases/'
paraphrases_dir = 'data/pattern_data/parsed'
cooccurrences_dir = 'data/output/spike_results/cooccurrences/'
lm_dir = 'data/output/predictions_lm/bert_lama/'


all_relations = []
for relation in glob('data/pattern_data/parsed/*.jsonl'):
    all_relations.append(relation.split('/')[-1].split('.')[0])


st.title('Explaining Model Success')
# st.write(pattern_id)

use_simple_cooccurrence_count = st.sidebar.checkbox('Use simple co-occurrence (by counts)', value=False)
if use_simple_cooccurrence_count:
    min_cooccurrence = st.sidebar.slider('minimum cooccurence', min_value=0, max_value=1000, value=100, step=10)
else:
    min_cooccurrence = None

only_correct_predictions = st.sidebar.checkbox('Only look at accurate predictions', value=False)

max_rank = st.sidebar.slider('maximum rank', min_value=0, max_value=20, value=5, step=1)


all_patterns = read_jsonl_file(relations_file)
relations2labels = {x['relation']: x['label'] for x in all_patterns}

relevants_relations = [x for x in all_relations if x in relations2labels.keys()]
relation_names = [f'{relations2labels[x]} ({x})' for x in relevants_relations]
selected_pattern = st.sidebar.radio('Choose Relation', relation_names, 15)
pattern_id = selected_pattern.split(' ')[-1][1:-1]


paraphrase_file = f'{paraphrases_dir}/{pattern_id}.jsonl'
lm_file = f'{lm_dir}/{pattern_id}_bert-large-cased.json'
cooccurrence_file = f'{cooccurrences_dir}/{pattern_id}.json'
memorization_file = f'{memorization_dir}/{pattern_id}.json'

relation_pattern = [x['template'] for x in all_patterns if x['relation'] == pattern_id][0].replace(' .', '.')
paraphrases = read_jsonl_file(paraphrase_file)
# if pattern_id == 'P449':
#     spike_pattern = "<>subject:Lost $was $aired $on object:[w={}]ABC."
# else:
print(relation_pattern)
print([x['pattern'] for x in paraphrases])
if relation_pattern not in [x['pattern'] for x in paraphrases]:
    relation_pattern = relation_pattern.replace('.', ' .')
spike_pattern = [x['pattern'] for x in paraphrases if x['pattern'] == relation_pattern]
if len(spike_pattern) == 0:
    st.write('pattern not supported yet')
    raise StopException
spike_pattern = spike_pattern[0]


lm_results = read_json_file(lm_file)
lm_predictions = get_lm_preds(lm_results[relation_pattern])

preference_bias = read_json_file(bias_file)[pattern_id]
cooccurrences = read_json_file(cooccurrence_file)
memorization = read_json_file(memorization_file)

pattern_data = get_items(memorization)

memorization_explained = explain_memorization(memorization, spike_pattern)
cooccurrence_explained = explain_cooccurrences(cooccurrences, min_cooccurrence, pattern_data, lm_results[relation_pattern],
                                               min_count_cooccurrence=min_cooccurrence is not None)
preference_bias_explained = explain_preference_bias(preference_bias, max_rank, pattern_data)
inclusion_explained = explain_subject_contains_object(pattern_data)

explanations = {}
for k, v in memorization_explained.items():
    explanations[k] = {**v,
                       **cooccurrence_explained[k],
                       **preference_bias_explained[k],
                       **inclusion_explained[k]
                       }

n_explanations = 0
explanation_type = defaultdict(int)
lm_correct_count = 0

table_data = []
for k, tuple_explanation in explanations.items():
    # excluding cases where the LM does not get the answer right
    if k not in lm_predictions or not lm_predictions[k]:
        success = False
    else:
        success = True
        lm_correct_count += 1
    if only_correct_predictions and (k not in lm_predictions or not lm_predictions[k]):
        continue

    found_explanation = False
    for specific_explanation, val in tuple_explanation.items():
        if val is not None:
            found_explanation = True
            explanation_type[specific_explanation] += 1
    if found_explanation:
        n_explanations += 1
    row = k.split('_')
    assert len(row) == 2
    row.extend([tuple_explanation['memorization'],
                tuple_explanation['cooccurences'],
                tuple_explanation['preference'],
                tuple_explanation['contains'],
                success
                ])

    table_data.append(row)

df = pd.DataFrame(table_data, columns=['subject', 'object', 'memorization', 'cooccurrences', 'preference', 'contains',
                                       'success'])
df = df.replace(np.nan, -1)
df = df.astype({"cooccurrences": int, "preference": int, "contains": int})
df = df.style.apply(highlight_errors, axis=1)


st.write('overall tuples:', len(table_data))
st.write('looking only at accurate predictions:', only_correct_predictions)
st.write('overall model success:', lm_correct_count / len(table_data))
st.write('managed to explain:', n_explanations, 'that is {:.1f}%'.format(100. * n_explanations / len(table_data)))
st.write('explanation by category:', explanation_type)

st.write(df)
