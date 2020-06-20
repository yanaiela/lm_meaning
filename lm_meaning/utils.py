import torch
from transformers import *
import string
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
