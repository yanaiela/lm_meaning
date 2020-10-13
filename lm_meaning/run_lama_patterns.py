import argparse
import json

import torch
from tqdm import tqdm
from transformers import pipeline, Pipeline
import wandb
from glob import glob
from typing import List, Dict
from lm_meaning import utils
from lm_meaning.run_pipeline import parse_prompt, build_model_by_name, run_query


def log_wandb(args):
    lm = args.lm

    config = dict(
        lm=lm
    )

    wandb.init(
        name=f'LAMA_lm_{lm}',
        project="memorization",
        tags=["lm"],
        config=config,
    )


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--lama_patterns", type=str, help="Path to lama patterns",
                       default="data/trex/data/relations.jsonl")
    parse.add_argument("--data_path", type=str, help="", default="/data/LAMA_data/TREx/")
    parse.add_argument("--pred_path", type=str, help="Path to store LM predictions for each prompt",
                       default="data/output/predictions_lm/lama/")
    parse.add_argument("--evaluate", action='store_true')
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--bs", type=int, default=50)
    parse.add_argument("--wandb", action='store_true')
    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    lama_patterns = utils.read_json_file(args.lama_patterns)
    rel2pattern = {x['relation']: x['template'] for x in lama_patterns}

    model_name = args.lm

    print('Language Models: {}'.format(model_name))
    model = build_model_by_name(model_name, args)

    # Load data
    for file_name in glob(args.data_path + '/*'):
        print(file_name)
        data = utils.read_json_file(file_name)
        pattern_name = file_name.split('/')[-1].split('.')[0]
        pattern = rel2pattern[pattern_name]
        results_dict = {}

        filtered_data, predictions = run_query(model, data, pattern, args.bs)
        results_dict[pattern] = {"data": filtered_data, "predictions": predictions}

        # Evaluate
        # if args.evaluate:
        #     accuracy = lm_eval(results_dict, args.lm)

        json.dump(results_dict, open(args.pred_path + '/' + pattern_name + '.json', "w"))


if __name__ == '__main__':
    main()
