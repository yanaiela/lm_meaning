import torch
import numpy as np
from tqdm import tqdm


def sentences2ids(sentences, tokenizer):
    tokenized_sentences = []

    for sentence in sentences:
        prefix, suffix = sentence.split("[MASK]")
        prefix_tokens = tokenizer.tokenize(prefix)
        suffix_tokens = tokenizer.tokenize(suffix)
        tokens = [tokenizer.cls_token] + prefix_tokens + [tokenizer.mask_token] + suffix_tokens + [tokenizer.sep_token]
        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
        target_idx = len(prefix_tokens) + 1
        tokenized_sentences.append(input_ids)

    return torch.cat(tokenized_sentences), target_idx


def get_predictions(tokenizer, lm_model, sentences, k=10):
    """
    Getting top k model prediction for sentences.
    Assuming sentences are of the same length
    :param tokenizer: model tokenizer
    :param lm_model: lm model
    :param sentences: a btach of sentences with the [MASK] argument
    :param k: top k most probable words
    :return: list of lists, corresponding to the sentence batch and top k words in each one
    """
    tokenized_sentences, target_idx = sentences2ids(sentences, tokenizer)
    prediction_scores = lm_model(tokenized_sentences)[0]
    token_scores = prediction_scores[:, target_idx].detach().cpu().numpy()
    best_k = (np.argsort(token_scores, axis=1))[:, -k:]
    sentences_best_k = []
    for top_per_sentence in best_k:
        best_k_tokens = tokenizer.convert_ids_to_tokens(top_per_sentence)

        # getting rid of roberta's byte-level tokens
        best_k_tokens = [x.replace('Ġ', '') for x in best_k_tokens]

        sentences_best_k.append(best_k_tokens[::-1])
    return sentences_best_k


def eval_query(tokenizer, lm_model, vals_dic, query, debug=False, bs=50, k=10, ignore_special_tokens=False):
    acc = 0.
    total_vals = len(vals_dic)

    batch_queries = []
    batch_answers = []
    for lemma, plural in tqdm(vals_dic.items()):
        if len(batch_queries) < bs:
            batch_queries.append(query.format(lemma))
            batch_answers.append(plural)
            continue

        if debug:
            print(batch_queries)
        try:
            top_k_per_ex = get_predictions(tokenizer, lm_model, batch_queries, k=k)
        except RuntimeError:
            total_vals -= bs
            batch_queries = []
            batch_answers = []
            print('issue')
            continue

        for top_k, y in zip(top_k_per_ex, batch_answers):
            if top_k[0] == y:
                acc += 1
            #             else:
            #                 print(y, top_k[:2])
            if top_k[0] in tokenizer.special_tokens_map.values():
                i = 1
                while True:
                    if top_k[i] in tokenizer.special_tokens_map.values():
                        continue
                    else:
                        if top_k[i] == y:
                            acc += 1
                        else:
                            break
                    i += 1
        batch_queries = []
        batch_answers = []

    if len(batch_queries) != 0:
        total_vals -= len(batch_queries)

    return acc / total_vals
