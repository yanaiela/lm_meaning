'''

#TODO 1. Find LAMA prompts
#TODO 2. Support more models
'''

import argparse
import utils
import lm_utils
import os
import logging
from transformers import *
import json


def parse_prompt(prompt, subject_label, object_label):
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)
    prompt = prompt.replace(OBJ_SYMBOL, object_label)
    return prompt


#
def build_model_by_name(lm, args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    from transformers import pipeline
    model = pipeline("fill-mask", model=lm)
    # nlp(f"Joey aired on {nlp.tokenizer.mask_token}.")
    return model, model.tokenizer, model.tokenizer.mask_token


def run_query(tokenizer, model, vals_dic, prompt, MASK_TOKEN):
    data = []

    # create the text prompt
    for sample in vals_dic:
        data.append({'prompt': parse_prompt(prompt, sample["sub_label"], MASK_TOKEN), 'answer': sample["obj_label"],
                     'sub_label': sample["sub_label"], 'obj_label': sample["obj_label"]})

    predictions = model([sample["prompt"] for sample in data])

    return data, predictions


def lm_eval(results_dict, lm):
    import pdb

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
        # pdb.set_trace()
        if cue[1] in cue_to_predictions[cue]:
            correct += 1

    print(correct * 1.0 / total)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--relation", type=str, help="Name of relation")
    parse.add_argument("--lm", type=str, help="comma separated list of language models", default="bert-base-uncased")
    parse.add_argument("--output_file_prefix", type=str, help="")
    parse.add_argument("--prompts", type=str, help="Path to templates for each prompt", default="/data/LAMA_data/TREx")
    parse.add_argument("--data_path", type=str, help="", default="/data/LAMA_data/TREx")
    parse.add_argument("--pred_path", type=str, help="Path to store LM predictions for each prompt",
                       default="./predictions_TREx/")
    parse.add_argument("--evaluate", action='store_true')
    parse.add_argument("--gpu", action='store_true')
    args = parse.parse_args()

    # Load data
    data_file = os.path.join(args.data_path, args.relation + '.jsonl')
    if not os.path.exists(data_file):
        raise ValueError('Relation "{}" does not exist in data.'.format(args.relation))
    data = utils.read_data(data_file)

    # Load prompts
    prompt_file = os.path.join(args.prompts, args.relation + '.jsonl')
    if not os.path.exists(prompt_file):
        raise ValueError('Relation "{}" does not exist in prompts.'.format(args.relation))
    prompts = utils.load_prompts(prompt_file)

    models_names = args.lm.split(",")

    print('Language Models: {}'.format(models_names))

    results_dict = {}
    for lm in models_names:
        model, tokenizer, mask_token = build_model_by_name(lm, args)

        results_dict[lm] = {}

        for prompt_id, prompt in enumerate(prompts):
            results_dict[lm][prompt] = []
            filtered_data, predictions = run_query(tokenizer, model, data, prompt, mask_token)
            results_dict[lm][prompt] = {"data": filtered_data, "predictions": predictions}

    # Evaluate

    if args.evaluate:
        accuracy = lm_eval(results_dict, args.lm)

    # Persist predictions
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)

    for lm in models_names:
        json.dump(results_dict[lm], open(args.pred_path + "/results_{}.json".format(args.lm), "w"))


if __name__ == '__main__':
    main()
