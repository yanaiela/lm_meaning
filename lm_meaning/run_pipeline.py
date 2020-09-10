import argparse
import json

import torch
from tqdm import tqdm
from transformers import pipeline
import wandb

from lm_meaning import utils


def log_wandb(args):
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    lm = args.lm

    config = dict(
        pattern=pattern,
        lm=lm
    )

    wandb.init(
        name=f'{pattern}_lm_{lm}',
        project="memorization",
        tags=["lm", pattern],
        config=config,
    )


def parse_prompt(prompt, subject_label, object_label):
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)
    prompt = prompt.replace(OBJ_SYMBOL, object_label)
    return prompt


# get mlm model to predict masked token.
def build_model_by_name(lm, args):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """

    device = args.gpu
    if not torch.cuda.is_available():
        device = -1

    model = pipeline("fill-mask", model=lm, device=device)
    return model


def run_query(pipeline_model, vals_dic, prompt, bs=20):
    data = []

    mask_token = pipeline_model.tokenizer.mask_token

    # create the text prompt
    for sample in vals_dic:
        data.append({'prompt': parse_prompt(prompt, sample["sub_label"], mask_token), 'answer': sample["obj_label"],
                     'sub_label': sample["sub_label"], 'obj_label': sample["obj_label"]})

    batched_data = []
    for i in range(0, len(data), bs):
        batched_data.append(data[i: i + bs])

    predictions = []
    for batch in tqdm(batched_data):
        preds = pipeline_model([sample["prompt"] for sample in batch])
        predictions.extend(preds)

    return data, predictions


def lm_eval(results_dict, lm):
    cue_to_predictions = {}

    for prompt in results_dict[lm]:
        for sample_ind, sample in enumerate(results_dict[lm][prompt]["data"]):
            cue = (sample["sub_label"], sample["obj_label"])
            if cue not in cue_to_predictions:
                cue_to_predictions[cue] = []

            cue_to_predictions[cue] += [results_dict[lm][prompt]["predictions"][sample_ind][0]["token_str"]]

    correct, total = 0, 0
    for cue in cue_to_predictions:
        total += 1
        if cue[1] in cue_to_predictions[cue]:
            correct += 1

    print(correct * 1.0 / total)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--output_file_prefix", type=str, help="")
    parse.add_argument("--patterns_file", type=str, help="Path to templates for each prompt", default="/data/LAMA_data/TREx")
    parse.add_argument("--data_file", type=str, help="", default="/data/LAMA_data/TREx/P449.jsonl")
    parse.add_argument("--pred_file", type=str, help="Path to store LM predictions for each prompt")
    parse.add_argument("--evaluate", action='store_true')
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--bs", type=int, default=50)
    parse.add_argument("--wandb", action='store_true')
    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    # Load data
    data = utils.read_data(args.data_file)

    # Load prompts
    prompts = utils.load_prompts(args.patterns_file)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    results_dict = {}
    model = build_model_by_name(model_name, args)

    results_dict[model_name] = {}

    for prompt_id, prompt in enumerate(prompts):
        results_dict[model_name][prompt] = []
        filtered_data, predictions = run_query(model, data, prompt, args.bs)
        results_dict[model_name][prompt] = {"data": filtered_data, "predictions": predictions}

    # Evaluate
    if args.evaluate:
        accuracy = lm_eval(results_dict, args.lm)

    json.dump(results_dict[model_name], open(args.pred_file, "w"))


if __name__ == '__main__':
    main()
