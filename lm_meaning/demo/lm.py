import re
from typing import List
from glob import glob

import pandas as pd
import streamlit as st
from transformers import pipeline

MODELS = [
    'bert-base-cased',
    'models/consistency/bert_base_cased/3_100/consitancy_bert-base-cased_100_P1376_P276_P31/checkpoint-279',

]

MASK = '[MASK]'


def get_predictions(sentence, lm_model, k=10) -> List:
    outputs = lm_model(sentence.replace('[MASK]', lm_model.tokenizer.mask_token))

    return [x['token_str'] for x in outputs]


@st.cache(allow_output_mutation=True)
def get_bert_models(model_name):
    model = pipeline("fill-mask", model=model_name, topk=100)

    return model


st.title("MLM Demo")

models_paths = []
for f in glob('models/nora/**/pytorch_model.bin', recursive=True):
    models_paths.append(f.rsplit('/', 1)[0])
models_paths = list(set(models_paths))

lm_models = list(MODELS) + models_paths

st.sidebar.title("Pre trained LMs")
used_models = []
for model in lm_models:
    # default checked model
    if model == 'bert-base-uncased':
        check = st.sidebar.checkbox(model, value=True)
    else:
        check = st.sidebar.checkbox(model)
    if check:
        used_models.append(model)

models = []
for m in used_models:
    m_model = get_bert_models(m)
    models.append(m_model)

text = st.text_input("Input Sentence ('[MASK]' for the masking token)", value="Input examples are [MASK]!")
# k = st.number_input("top_k", min_value=1, max_value=100, value=10, step=1)

st.subheader('LM predictions')

if MASK not in text:
    st.text(f'the "{MASK}" must appear in the text')
else:
    if text != '':
        progress_bar = st.progress(0)

        num_mask = text.count(MASK)
        model_predictions = []

        # model_predictions
        for ind, model in enumerate(models):
            preds = get_predictions(text, model)
            model_predictions.append(preds)
            progress_bar.progress(int((float(ind + 1) / len(models)) * 100))
        progress_bar.progress(100)
        progress_bar.empty()

        mask_indices = [m.start() for m in re.finditer('\[MASK\]', text)]
        # markdown_text = text[:mask_ind] + '**' + MASK + '**' + text[mask_ind + len(MASK):]
        markdown_text = text.replace('[MASK]', '\[MASK\]')
        # st.write(f'#{ind + 1} mask:' + markdown_text)
        dict_data = {}
        for model, answers in zip(used_models, model_predictions):
            dict_data[model] = answers
        df = pd.DataFrame(dict_data)

        st.dataframe(df, width=1000, height=1000)
