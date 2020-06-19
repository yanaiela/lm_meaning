import pytest
import numpy as np
from lm_meaning.evaluation.lm_predict import split_data2batches, sentences2ids

from transformers import AutoTokenizer


def generate_examples(n, sen_len):
    random_data = []
    for i in range(n):
        ex = {'prompt': '', 'answer': '',
              'tokenized_sentence': np.random.randint(0, 100, sen_len),
              'masked_ind': np.random.randint(0, 100)}
        random_data.append(ex)
    return random_data


def test_split_data2batches():
    data_10 = generate_examples(5, 10)
    data_8 = generate_examples(5, 8)

    data = data_8 + data_10
    batched_data = split_data2batches(data, 5)

    print(batched_data)
    for batch in batched_data:
        first_len = len(batch[0]['tokenized_sentence'])
        assert len(batch) <= 5
        for ex_batch in batch:
            assert len(ex_batch['tokenized_sentence']) == first_len


def test_tokenization():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    sentence_example = "Hello [MASK]."
    tokenized_sentences, masked_indices = sentences2ids([sentence_example], tokenizer)
    assert len(tokenized_sentences) == len(masked_indices)

    tokens = tokenizer.tokenize(sentence_example)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    assert len(tokens) == len(tokenized_sentences[0])
    ids = tokenizer.convert_tokens_to_ids(tokens)
    assert tokenized_sentences[0] == ids


def test_tokenization_uneven():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    sentence_example1 = "Hello [MASK]."
    sentence_example2 = "Hello there [MASK]."
    tokenized_sentences, masked_indices = sentences2ids([sentence_example1, sentence_example2], tokenizer)
    assert len(tokenized_sentences) == len(masked_indices)
