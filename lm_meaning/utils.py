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


def filter_vals(in_dic):
    filter_inflections = {}

    for k, v in tqdm(in_dic.items()):
        tok_v = tokenizer.tokenize(v)
        tok_k = tokenizer.tokenize(k)

        if len(k) == 1:
            continue
        if len(tok_v) != 1 or len(tok_k) != 1:
            continue

        if not all([x in (string.ascii_lowercase + string.ascii_uppercase) for x in k]):
            #             print(k)
            continue

        filter_inflections[k] = v

    return filter_inflections
