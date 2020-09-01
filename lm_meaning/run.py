import argparse
import json
import os

from lm_meaning.utils import filter_data_fields, read_data, load_prompts
from lm_meaning.lm_utils import build_model_by_name, run_query, lm_eval


def query_lm(model_names, prompts, data, use_gpu):
    results_dict = {}
    for lm in model_names:
        model, tokenizer, mask_token = build_model_by_name(lm, use_gpu)

        results_dict[lm] = {}

        for prompt_id, prompt in enumerate(prompts):
            results_dict[lm][prompt] = []
            predictions = run_query(tokenizer, model, data, prompt, mask_token, use_gpu=use_gpu)
            results_dict[lm][prompt] = {"data": filter_data_fields(data), "predictions": predictions}
    return results_dict


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
    data = read_data(data_file)

    # Load prompts
    prompt_file = os.path.join(args.prompts, args.relation + '.jsonl')
    if not os.path.exists(prompt_file):
        raise ValueError('Relation "{}" does not exist in prompts.'.format(args.relation))
    prompts = load_prompts(prompt_file)

    models_names = args.lm.split(",")

    print('Language Models: {}'.format(models_names))

    results_dict = query_lm(models_names, prompts, data, args.gpu)

    # Evaluate

    if args.evaluate:
        accuracy = lm_eval(results_dict, args.lm)

    # Persist predictions
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)

    for lm in models_names:
        json.dump(results_dict[lm], open(args.pred_path+"/{}_{}_origin_new.json".format(args.relation, args.lm), "w"))


if __name__ == '__main__':
    main()
