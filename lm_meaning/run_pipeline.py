import argparse
import json
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import pipeline, Pipeline, BertForMaskedLM, BertTokenizer
import wandb

from typing import List, Dict
from lm_meaning import utils


def log_wandb(args):
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    lm = args.lm

    config = dict(
        pattern=pattern,
        lm=lm
    )

    # entailment_bert-large-cased-whole-word-masking_100_4_2_P176-P30-P39-P127
    if 'entailment' in lm:
        model_args = lm.split('_')
        config['ft_type'] = model_args[0]
        config['model_name'] = model_args[1]
        config['n_tuples'] = model_args[2]
        config['n_graphs'] = model_args[3]
        config['epoch'] = model_args[4]
        config['graphs_trained'] = model_args[5]


    wandb.init(
        name=f'{pattern}_lm_{lm}',
        project="memorization",
        tags=["lm", pattern],
        config=config,
    )


def parse_prompt(prompt: str, subject_label: str, object_label: str) -> str:
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)\
                   .replace(OBJ_SYMBOL, object_label)
    return prompt


# get mlm model to predict masked token.
def build_model_by_name(lm: str, args) -> Pipeline:
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """

    device = args.gpu
    if not torch.cuda.is_available():
        device = -1

    if 'consistancy' in lm:
        model = BertForMaskedLM.from_pretrained(lm)
        tokenizer = BertTokenizer.from_pretrained("bert-large-cased-whole-word-masking")
        model = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device, top_k=100)
    else:
        model = pipeline("fill-mask", model=lm, device=device, top_k=100)

    return model


def get_original_token(tokenized_obj, possible_objects, tokenizer):
    for obj in possible_objects:
        if tokenizer.tokenize(obj)[0] == tokenized_obj:
            return obj.strip()
    return None


def tokenize_results(results, pipeline_model, possible_objects):
    if pipeline_model.model.config.model_type in ['roberta']:
        preds_tokenized = []
        for example in results:
            example_tokenized = []
            for ans in example:
                ans_copy = deepcopy(ans)
                # tokenized_obj_ans = pipeline_model.tokenizer.convert_tokens_to_string(ans['token_str']).strip()
                #original_obj_ans = get_original_token(ans['token_str'], possible_objects, pipeline_model.tokenizer)
                original_obj_ans = pipeline_model.tokenizer.convert_tokens_to_string(ans['token_str'])
                #assert original_obj_ans is not None, "did not find object in tokenized objects"
                ans_copy['token_str'] = original_obj_ans

                example_tokenized.append(ans_copy)
            preds_tokenized.append(example_tokenized)
        return preds_tokenized
    else:
        return results


def run_query(pipeline_model: Pipeline, vals_dic: List[Dict], prompt: str, possible_objects: List[str], bs: int = 20)\
        -> (List[Dict], List[Dict]):
    data = []

    mask_token = pipeline_model.tokenizer.mask_token

    # create the text prompt
    for sample in vals_dic:
        data.append({'prompt': parse_prompt(prompt, sample["sub_label"], mask_token),
                     'sub_label': sample["sub_label"], 'obj_label': sample["obj_label"]})

    batched_data = []
    for i in range(0, len(data), bs):
        batched_data.append(data[i: i + bs])

    predictions = []
    for batch in tqdm(batched_data):
        preds = pipeline_model([sample["prompt"] for sample in batch], targets=possible_objects)
        # pipeline_model returns a list in case there is only 1 item to predict (in contrast to list of lists)
        if len(batch) == 1:
            preds = [preds]
        tokenized_preds = tokenize_results(preds, pipeline_model, possible_objects)
        predictions.extend(tokenized_preds)

    data_reduced = []
    for row in data:
        if pipeline_model.model.config.model_type in ['albert']:
            data_reduced.append({'sub_label': row['sub_label'],
                                 'obj_label': pipeline_model.tokenizer.tokenize(row['obj_label'])[0]})
        elif pipeline_model.model.config.model_type in ['roberta']:
            data_reduced.append({'sub_label': row['sub_label'],
                                 'obj_label': ' ' + row['obj_label']})
        else:
            data_reduced.append({'sub_label': row['sub_label'], 'obj_label': row['obj_label']})

    preds_reduced = []
    for top_k in predictions:
        vals = []
        for row in top_k:
            vals.append({'score': row['score'], 'token': row['token'], 'token_str': row['token_str']})
        preds_reduced.append(vals)

    return data_reduced, preds_reduced


def lm_eval(results_dict: Dict, lm: str):
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
    parse.add_argument("--lm", type=str, help="name of the used masked language model", default="bert-large-cased")
    parse.add_argument("--output_file_prefix", type=str, help="")
    parse.add_argument("--patterns_file", type=str, help="Path to templates for each prompt", default="/data/LAMA_data/TREx")
    parse.add_argument("--data_file", type=str, help="", default="/data/LAMA_data/TREx/P449.jsonl")
    parse.add_argument("--pred_file", type=str, help="Path to store LM predictions for each prompt")
    parse.add_argument("--evaluate", action='store_true')
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--bs", type=int, default=100)
    parse.add_argument("--wandb", action='store_true')
    parse.add_argument("--no_subj", type=bool, default=False)
    parse.add_argument("--use_targets", action='store_true', default=False, help="use the set of possible objects"
                                                                                 "from the data as the possible"
                                                                                 "candidates")

    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    # Load data
    if args.no_subj:
        data = [{"sub_label": "", "obj_label": ""}]
    else:
        data = utils.read_jsonl_file(args.data_file)

    if args.use_targets:
        all_objects = list(set([x['obj_label'] for x in data]))
        if 'roberta' in args.lm or 'albert' in args.lm:
            all_objects = [' ' + x for x in all_objects]
    else:
        all_objects = None

    # Load prompts
    prompts = utils.load_prompts(args.patterns_file)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    results_dict = {}
    model = build_model_by_name(model_name, args)

    results_dict[model_name] = {}

    for prompt_id, prompt in enumerate(prompts):
        results_dict[model_name][prompt] = []
        filtered_data, predictions = run_query(model, data, prompt, all_objects, args.bs)
        results_dict[model_name][prompt] = {"data": filtered_data, "predictions": predictions}

    # Evaluate
    if args.evaluate:
        accuracy = lm_eval(results_dict, args.lm)

    json.dump(results_dict[model_name], open(args.pred_file, "w"))


if __name__ == '__main__':
    main()
