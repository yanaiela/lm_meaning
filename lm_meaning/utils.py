import json
import string
from typing import List, Dict

from transformers import *

model_name = 'roberta-large'

model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()


def filter_vocab(items):
    filtered_items = []

    for item in items:
        tok_v = tokenizer.tokenize(item)

        if len(tok_v) != 1:
            continue

        if not all([x in (string.ascii_lowercase + string.ascii_uppercase) for x in tok_v[0]]):
            continue

        filtered_items.append(item[0])

    return filtered_items


def read_data(filename: str) -> List[Dict]:

    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)

    return dataset


def filter_data_fields(data):
    return [{"sub_label": sample["sub_label"], "obj_label":sample["obj_label"]} for sample in data]


def parse_prompt(prompt: str, subject_label: str, object_label: str) -> str:
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)
    prompt = prompt.replace(OBJ_SYMBOL, object_label)
    return prompt


def load_prompts(filename: str) -> List[str]:
    prompts = []
    with open(filename, 'r') as fin:
        for row in fin:
            l = json.loads(row)
            prompt = l['pattern']
            prompts.append(prompt)
    return prompts



