'''

#TODO 1. Find LAMA prompts
#TODO 2. Support more models
'''

import argparse
import utils
import lm_utils
import os
import logging
from transformers import BertForMaskedLM, AutoTokenizer, pipeline
import json


#
def build_model_by_name(lm, args, verbose=True):
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
        if args.gpu:
            model.cuda()
        model.eval()
    else:
        model = MODEL_NAME_TO_CLASS[model_type]
        tokenizer = model.tokenizer
    return model, tokenizer, masked_tokens[model_type]


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--relation", type=str, help="Name of relation")
    parse.add_argument("--lm", type=str, help="comma separated list of language models", default="bert-base-uncased")
    parse.add_argument("--output_file_prefix", type=str, help="")
    parse.add_argument("--prompts", type=str, help="Path to templates for each prompt", default="data/lm_relations/")
    parse.add_argument("--data_path", type=str, help="", default="data/LAMA_data/TREx")
    parse.add_argument("--pred_path", type=str, help="Path to store LM predictions for each prompt",
                       default="data/predictions_lm/")
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
            predictions = lm_utils.run_query(tokenizer, model, data, prompt, mask_token, use_gpu=args.gpu)
            results_dict[lm][prompt] = {"data": utils.filter_data_fields(data), "predictions": predictions}

    # Evaluate

    if args.evaluate:
        accuracy = lm_utils.lm_eval(results_dict, args.lm)

    # Persist predictions
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)

    for lm in models_names:
        json.dump(results_dict[lm], open(args.pred_path+"/{}_{}_origin.json".format(args.relation, args.lm), "w"))


if __name__ == '__main__':
    main()
