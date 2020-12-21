import re
from typing import List

import pandas as pd
import streamlit as st
from transformers import pipeline

MODELS = {
    'bert-base-cased': 'bert-base-cased'

}

MASK = '[MASK]'


def get_predictions(sentence, lm_model, k=10) -> List:
    outputs = lm_model(sentence.replace('[MASK]', lm_model.tokenizer.mask_token))

    return [x['token_str'] for x in outputs]


@st.cache(allow_output_mutation=True)
def get_bert_models(model_name):
    model = pipeline("fill-mask", model=model_name, topk=100)

    return model


st.title("MLM Demo")

lm_models = list(MODELS.keys())

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
    m_model = get_bert_models(MODELS[m])
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
