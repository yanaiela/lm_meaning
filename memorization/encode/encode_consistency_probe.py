import argparse
import json
from collections import defaultdict
from typing import List, Dict

import wandb

from memorization.encode.run_pipeline import build_model_by_name, run_query
from memorization.encode.utils import read_jsonl_file


def log_wandb(args):
    pattern = args.data_file.split('/')[-1].split('.')[0]
    lm = args.lm

    rand = str(args.random_weights)

    config = dict(
        pattern=pattern,
        lm=lm,
        random_weights=args.random_weights
    )

    wandb.init(
        entity='consistency',
        name=f'encode_{pattern}_{lm}_rand:{rand}',
        project="memorization",
        tags=[pattern, 'encode'],
        config=config,
    )


def get_first_object(preds, possible_objects):
    for row in preds:
        token = row['token_str']
        if token in possible_objects:
            return token
    return ''


def parse_lm_results(lm_results: Dict, possible_objects: List[str]) -> Dict:
    output_dic = defaultdict(dict)
    c = 0
    for pattern, dic in lm_results.items():
        for data, preds in zip(dic['data'], dic['predictions']):
            subj = data['sub_label']
            obj = data['obj_label']
            first_object = get_first_object(preds, possible_objects)
            output_dic[pattern][subj] = (first_object, obj)
    return output_dic


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm", type=str, help="name of the used masked language model", default="bert-base-cased")
    parse.add_argument("--data_file", type=str, help="", default="data/trex_lms_vocab/P449.jsonl")
    parse.add_argument("-patterns", "--patterns", type=str, help="graph file",
                       default="data/pattern_data/graphs_tense_json/P449.jsonl")
    parse.add_argument("--out", type=str, help="output folder",
                       default="data/output/predictions_lm/P449_bert-base-cased.jsonl")
    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--bs", type=int, default=200)
    parse.add_argument("--wandb", action='store_true')
    parse.add_argument("--no_subj", type=bool, default=False)
    parse.add_argument("--baseline", action='store_true', default=False)
    parse.add_argument("--use_targets", action='store_true', default=False, help="use the set of possible objects"
                                                                                 "from the data as the possible"
                                                                                 "candidates")
    parse.add_argument('--random_weights', default=False, type=lambda x: (str(x).lower() == 'true'),
                       help="randomly initialize the models' weights")

    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    # Load data
    if args.no_subj:
        data = [{"sub_label": "", "obj_label": ""}]
    else:
        data = read_jsonl_file(args.data_file)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    model = build_model_by_name(model_name, args)

    patterns = read_jsonl_file(args.patterns)

    subj_obj = {}
    for row in data:
        subj_obj[row['sub_label']] = row['obj_label']

    # Load prompts
    prompts = [x['pattern'] for x in patterns]

    if args.use_targets:
        all_objects = list(set([x['obj_label'] for x in data]))
        # if 'roberta' in args.lm or 'albert' in args.lm:
        if 'roberta' in args.lm:
            all_objects = [' ' + x for x in all_objects]
        elif 'albert' in args.lm:
            all_objects = [model.tokenizer.tokenize(x)[0] for x in all_objects]
        elif 'google' in args.lm:
            all_objects = [x.lower() for x in all_objects]
    else:
        all_objects = None

    results_dict = {}
    for prompt_id, prompt in enumerate(prompts):
        results_dict[prompt] = []
        filtered_data, predictions = run_query(model, data, prompt, all_objects, args.bs)
        results_dict[prompt] = {"data": filtered_data, "predictions": predictions}

    lm_results = parse_lm_results(results_dict, all_objects)

    if 'models' in model_name or 'nyu' in model_name or 'google' in model_name:
        model_name = model_name.replace('/', '_')
    with open(args.out.replace(args.lm, model_name), 'w') as f:
        json.dump(lm_results, f)


if __name__ == '__main__':
    main()
