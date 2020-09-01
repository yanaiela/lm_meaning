from collections import defaultdict

import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm
from transformers import BertForMaskedLM, AutoTokenizer, RobertaForMaskedLM


def build_model_by_name(lm, use_gpu, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    model_type = lm.split("-")[0]
    MODEL_NAME_TO_CLASS = dict(
        bert=BertForMaskedLM,
        roberta=RobertaForMaskedLM,
    )
    masked_tokens = dict(
        bert="[MASK]",
        roberta="[MASK]"
    )
    if model_type not in MODEL_NAME_TO_CLASS:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)

    if model_type == "bert" or model_type=="roberta":
        model = MODEL_NAME_TO_CLASS[model_type].from_pretrained(lm)
        tokenizer = AutoTokenizer.from_pretrained(lm)
        if use_gpu:
            model.cuda()
        model.eval()
    else:
        model = MODEL_NAME_TO_CLASS[model_type]
        tokenizer = model.tokenizer
    return model, tokenizer, masked_tokens[model_type]


def parse_prompt(prompt, subject_label, object_label):
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)
    prompt = prompt.replace(OBJ_SYMBOL, object_label)
    return prompt


def sentences2ids(sentences, tokenizer, mask_token="[MASK]"):
    tokenized_sentences = []
    masked_word_indices = []

    for sentence in sentences:
        prefix, suffix = sentence.split(mask_token)
        prefix_tokens = tokenizer.tokenize(prefix.strip())
        suffix_tokens = tokenizer.tokenize(suffix.strip())
        tokens = [tokenizer.cls_token] + prefix_tokens + [tokenizer.mask_token] + suffix_tokens + [tokenizer.sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        target_idx = len(prefix_tokens) + 1
        tokenized_sentences.append(input_ids)
        masked_word_indices.append(target_idx)

    return tokenized_sentences, masked_word_indices


def get_predictions(tokenizer, lm_model, tokenized_sentences, target_idx, use_gpu=False, k=10):
    """
    Getting top k model prediction for sentences.
    Assuming sentences are of the same length
    :param tokenizer: model tokenizer
    :param lm_model: lm model
    :param tokenized_sentences: a batch of sentences with the [MASK] argument
    :param k: top k most probable words
    :return: list of lists, corresponding to the sentence batch and top k words in each one
    """
    queries = torch.tensor(tokenized_sentences)
    if use_gpu:
        queries = queries.cuda()
    prediction_scores = lm_model(queries)[0]
    token_scores = torch.stack([x[y] for x, y in zip(prediction_scores, target_idx)]).detach().cpu().numpy()
    best_k = (np.argsort(token_scores, axis=1))[:, -k:]
    sentences_best_k = []
    for top_per_sentence in best_k:
        best_k_tokens = tokenizer.convert_ids_to_tokens(top_per_sentence)

        # "normalizing" the tokens to 'standard' strings
        best_k_tokens = ["".join(tokenizer.convert_tokens_to_string(x).strip().split()) for x in best_k_tokens]
        best_k_tokens = [x.replace('Ġ', '') for x in best_k_tokens]
        sentences_best_k.append(best_k_tokens[::-1])
    return sentences_best_k


def get_most_similar(weights, vec):
    distances = distance.cdist(np.array([vec]), weights, "cosine")[0]
    most_similar = np.argsort(distances)
    # 0 index is the same vector, returning the next most similar vector
    return most_similar[1]


def lm_baseline(tokenizer, lm_model, vals_dic):

    word_embeddings = lm_model.get_input_embeddings().weight.detach().numpy()

    acc = 0.
    for k, v in vals_dic.items():
        token_id = tokenizer.convert_tokens_to_ids(k)
        most_similar_id = get_most_similar(word_embeddings, word_embeddings[token_id])
        most_similar_token = tokenizer.convert_ids_to_tokens(int(most_similar_id))
        most_similar_token_string = tokenizer.convert_tokens_to_string([most_similar_token]).strip()
        if v == most_similar_token_string:
            acc += 1

    total_vals = len(vals_dic)
    return acc / total_vals


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


def data2batches(prompt, vals_dic, tokenizer, batch_size, MASK_TOKEN="[MASK]"):
    data = []

    # create the text prompt
    for sample in vals_dic:
        data.append({'prompt': parse_prompt(prompt, sample["sub_label"], MASK_TOKEN), 'answer': sample["obj_label"]})

    tokenized_sentences, target_indices = sentences2ids([x['prompt'] for x in data], tokenizer, mask_token=MASK_TOKEN)

    for example, tokenized_sentence, target_index in zip(data, tokenized_sentences, target_indices):
        example.update({'tokenized_sentence': tokenized_sentence,
                        'masked_ind': target_index})

    batched_data = split_data2batches(data, batch_size)
    return batched_data


def run_query(tokenizer, lm_model, vals_dic, prompt, mask_token="[MASK]", use_gpu=False, debug=False, bs=50, k=10,
              ignore_special_tokens=False):

    batched_data = data2batches(prompt, vals_dic, tokenizer, bs, mask_token)

    predictions = []

    for batch in tqdm(batched_data):

        tokenized_sentence = [x['tokenized_sentence'] for x in batch]
        mask_indices = [x['masked_ind'] for x in batch]
        answers = [x['answer'] for x in batch]

        top_k_per_ex = get_predictions(tokenizer, lm_model, tokenized_sentence, mask_indices, use_gpu=use_gpu, k=k)

        predictions += top_k_per_ex

    return predictions


def lm_eval(results_dict, lm):
    cue_to_predictions = {}

    for prompt in results_dict[lm]:
        for sample_ind, sample in enumerate(results_dict[lm][prompt]["data"]):
            cue = (sample["sub_label"], sample["obj_label"])
            if cue not in cue_to_predictions:
                cue_to_predictions[cue] = []
            cue_to_predictions[cue] += results_dict[lm][prompt]["predictions"][sample_ind]

    correct, total = 0, 0
    for cue in cue_to_predictions:
        total += 1
        if cue[1] in cue_to_predictions[cue]:
            correct += 1

    print(correct * 1.0 / total)
