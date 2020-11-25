import argparse
from typing import List

from transformers import AutoTokenizer

from lm_meaning.utils import read_jsonl_file, write_jsonl_file

import wandb


def log_wandb(args):
    models = args.model_names
    pattern = args.in_data.split('/')[-1].split('.')[0]
    config = dict(
        pattern=pattern,
        models=models,
    )

    wandb.init(
        name=f'{pattern}_filter_oov',
        project="memorization",
        tags=[pattern],
        config=config,
    )


def get_tokenizer_by_name(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_tokenizers(model_names: List[str]):
    tokenizers = [get_tokenizer_by_name(x) for x in model_names]
    return tokenizers


def filter_oov(data, tokenizer):

    filt_data = []

    for row in data:
        subj = row['sub_label']
        obj = row['obj_label']
        if len(tokenizer.tokenize(f' {obj}')) != 1:
            continue
        filt_data.append({'sub_label': subj, 'obj_label': obj,
                          'uuid': row['uuid']})
    return filt_data


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--in_data", type=str, help="jsonl file",
                       default="data/trex/data/TREx/P1001.jsonl")
    parse.add_argument("--model_names", type=str, help="model type (out of MLM from huggingface)",
                       default="roberta-base,bert-base-cased")
    parse.add_argument("--out_file", type=str, help="output jsonl file path",
                       default="data/trex_lms_vocab/P1001.jsonl")

    args = parse.parse_args()
    log_wandb(args)

    data = read_jsonl_file(args.in_data)
    original_length = len(data)

    model_names = args.model_names.split(',')
    tokenizers = get_tokenizers(model_names)

    for tokenizer_name, tokenizer in zip(model_names, tokenizers):
        before = len(data)
        data = filter_oov(data, tokenizer)
        after = len(data)
        print(f'{tokenizer_name} filtered out {before - after} examples')

    wandb.run.summary['original_length'] = original_length
    wandb.run.summary['filtered_length'] = len(data)
    write_jsonl_file(data, args.out_file)


if __name__ == '__main__':
    main()
