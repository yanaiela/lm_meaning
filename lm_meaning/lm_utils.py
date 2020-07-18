import torch
import numpy as np
from transformers import *

def get_tokenized_queries(sentence, tokenizer):
    print(sentence)

    prefix, suffix = sentence.split("[MASK]")
    prefix_tokens = tokenizer.tokenize(prefix)
    suffix_tokens = tokenizer.tokenize(suffix)
    tokens = [tokenizer.cls_token] + prefix_tokens + [tokenizer.mask_token] + suffix_tokens + [tokenizer.sep_token]
    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    target_idx = len(prefix_tokens) + 1



    return input_ids, target_idx


def get_predictions(sentence, model, tokenizer, k=1):
    tokenized_sentence, target_idx = get_tokenized_queries(sentence, tokenizer)

    #     print(tokenized_sentences)
    prediction_scores = model(tokenized_sentence)[0]


    token_scores = prediction_scores[:, target_idx].detach().cpu()
    best_k = torch.argsort(token_scores, axis=1, descending=True).reshape(-1)[:k].cpu().numpy()    #     print(best_k)

    best_k_tokens = tokenizer.convert_ids_to_tokens(best_k)


    return best_k_tokens