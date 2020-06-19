import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def sentences2ids(sentences, tokenizer):
    tokenized_sentences = []
    masked_word_indices = []

    for sentence in sentences:
        prefix, suffix = sentence.split("[MASK]")
        prefix_tokens = tokenizer.tokenize(prefix)
        suffix_tokens = tokenizer.tokenize(suffix)
        tokens = [tokenizer.cls_token] + prefix_tokens + [tokenizer.mask_token] + suffix_tokens + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        target_idx = len(prefix_tokens) + 1
        tokenized_sentences.append(input_ids)
        masked_word_indices.append(target_idx)

    return tokenized_sentences, masked_word_indices


def get_predictions(tokenizer, lm_model, tokenized_sentences, target_idx, k=10):
    """
    Getting top k model prediction for sentences.
    Assuming sentences are of the same length
    :param tokenizer: model tokenizer
    :param lm_model: lm model
    :param sentences: a btach of sentences with the [MASK] argument
    :param k: top k most probable words
    :return: list of lists, corresponding to the sentence batch and top k words in each one
    """
    # tokenized_sentences, target_idx = sentences2ids(sentences, tokenizer)
    prediction_scores = lm_model(torch.tensor(tokenized_sentences))[0]
    # token_scores = prediction_scores[:, target_idx].detach().cpu().numpy()
    token_scores = torch.stack([x[y] for x, y in zip(prediction_scores, target_idx)]).detach().cpu().numpy()
    best_k = (np.argsort(token_scores, axis=1))[:, -k:]
    sentences_best_k = []
    for top_per_sentence in best_k:
        best_k_tokens = tokenizer.convert_ids_to_tokens(top_per_sentence)

        # getting rid of roberta's byte-level tokens
        best_k_tokens = [x.replace('Ġ', '') for x in best_k_tokens]

        sentences_best_k.append(best_k_tokens[::-1])
    return sentences_best_k


def split_data2batches(data_list, max_batch_size):
    len_buckets = defaultdict(list)

    for example in data_list:
        tokenized_sentence = example['tokenized_sentence']
        len_buckets[len(tokenized_sentence)].append(example)

    batched_data = []
    for sen_len, examples in len_buckets.items():
        same_size_bucket = []
        for example in examples:
            if len(same_size_bucket) >= max_batch_size:
                batched_data.append(same_size_bucket)
                same_size_bucket = []
            same_size_bucket.append(example)
        if len(same_size_bucket) > 0:
            batched_data.append(same_size_bucket)
    return batched_data


def data2batches(query, vals_dic, tokenizer, batch_size):
    data = []

    # create the text prompt
    for k, v in vals_dic.items():
        data.append({'prompt': query.format(k), 'answer': v})

    tokenized_sentences, target_indices = sentences2ids([x['prompt'] for x in data], tokenizer)

    for example, tokenized_sentence, target_index in zip(data, tokenized_sentences, target_indices):
        example.update({'tokenized_sentence': tokenized_sentence,
                        'masked_ind': target_index})

    batched_data = split_data2batches(data, batch_size)
    return batched_data


def eval_query(tokenizer, lm_model, vals_dic, query, debug=False, bs=50, k=10, ignore_special_tokens=False):
    acc = 0.
    total_vals = len(vals_dic)

    batched_data = data2batches(query, vals_dic, tokenizer, bs)

    for batch in batched_data:
        tokenized_sentence = [x['tokenized_sentence'] for x in batch]
        mask_indices = [x['masked_ind'] for x in batch]
        answers = [x['answer'] for x in batch]
        top_k_per_ex = get_predictions(tokenizer, lm_model, tokenized_sentence, mask_indices, k=k)

        for top_k, y in zip(top_k_per_ex, answers):
            if top_k[0] == y:
                acc += 1
            if ignore_special_tokens and top_k[0] in tokenizer.special_tokens_map.values():
                i = 1
                while True:
                    if top_k[i] in tokenizer.special_tokens_map.values():
                        i += 1
                    else:
                        if top_k[i] == y:
                            acc += 1
                        break

    return acc / total_vals
