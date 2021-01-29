import argparse
from typing import Dict
from collections import Counter
import wandb
import json
from lm_meaning.evaluation.consistency_probe import analyze_results, analyze_graph, parse_lm_results
from lm_meaning.run_pipeline import build_model_by_name, run_query
from lm_meaning.utils import read_graph, read_jsonl_file


def log_wandb(args):
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    lm = args.lm

    if args.baseline:
        lm = 'majority-baseline'

    config = dict(
        pattern=pattern,
        lm=lm
    )

    if 'consistency' in lm:
        params = lm.split('consistency_')[-1].split('/')[0]
        model_args = params.split('_')

        config['loss_strategy'] = model_args[0]
        config['n_tuples'] = model_args[1]
        config['n_graphs'] = model_args[2]
        config['rels_train'] = model_args[3]
        config['origin_lm'] = model_args[4]
        config['loss'] = model_args[5]
        config['vocab'] = model_args[6]
        config['wiki'] = model_args[7]
        config['lama_train'] = model_args[8]
        config['wiki_consistent_train_ratio'] = model_args[9]
        config['consistent_loss_ratio'] = model_args[10]
        config['additional_notes'] = model_args[11]

        checkpoint = lm.split('checkpoint-')[-1].split('/')[0]
        config['checkpoint'] = checkpoint

        model_name = lm.split('/checkpoint-')[0]
        config['model_name'] = model_name


    wandb.init(
        entity='consistency',
        name=f'{pattern}_consistency_probe_{lm}',
        project="consistency",
        tags=[pattern, 'probe'],
        config=config,
    )


def evaluate_lama(pattern: str, lm_results: Dict):

    points = 0
    data, predictions = lm_results[pattern]['data'], lm_results[pattern]['predictions']
    for datum, preds in zip(data, predictions):
        subj = datum['sub_label']
        obj = datum['obj_label']
        pred_obj = preds[0]['token_str']
        if pred_obj == obj:
            points += 1
    return points / len(data)


def group_score_lama_eval(lm_results: Dict):
    patterns = list(lm_results.keys())

    points = 0
    data = lm_results[patterns[0]]['data']
    for datum_ind, datum in enumerate(data):
        obj = datum['obj_label']
        consistent_true = True
        for pattern in patterns:
            preds = lm_results[pattern]['predictions'][datum_ind]
            if preds[0]['token_str'] != obj:
                consistent_true = False
                break

        if consistent_true:
            points += 1

    return points / len(data)


def create_majority_baseline(data):
    data_reduced = []
    for row in data:
        data_reduced.append({'sub_label': row['sub_label'], 'obj_label': row['obj_label']})

    objs = [x['obj_label'] for x in data]
    most_common = Counter(objs).most_common()[0][0]
    preds_reduced = []
    for _ in data:
        vals = [{'score': 1, 'token_str': most_common}]
        preds_reduced.append(vals)
    return data_reduced, preds_reduced


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--lm", type=str, help="name of the used masked language model", default="bert-base-uncased")
    parse.add_argument("--patterns_file", type=str, help="Path to templates for each prompt",
                       default="data/pattern_data/parsed")
    parse.add_argument("--data_file", type=str, help="", default="data/trex_lms_vocab/P449.jsonl")
    parse.add_argument("-graph", "--graph", type=str, help="graph file",
                       default="data/pattern_data/graphs/P449.graph")

    parse.add_argument("--gpu", type=int, default=-1)
    parse.add_argument("--bs", type=int, default=200)
    parse.add_argument("--wandb", action='store_true')
    parse.add_argument("--no_subj", type=bool, default=False)
    parse.add_argument("--baseline", action='store_true', default=False)
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
        data = read_jsonl_file(args.data_file)

    model_name = args.lm

    print('Language Models: {}'.format(model_name))

    model = build_model_by_name(model_name, args)

    patterns_graph = read_graph(args.graph)

    subj_obj = {}
    for row in data:
        subj_obj[row['sub_label']] = row['obj_label']

    # Load prompts
    prompts = [x.lm_pattern for x in list(patterns_graph.nodes)]

    if args.use_targets:
        all_objects = list(set([x['obj_label'] for x in data]))
        # if 'roberta' in args.lm or 'albert' in args.lm:
        if 'roberta' in args.lm:
            all_objects = [' ' + x for x in all_objects]
        elif 'albert' in args.lm:
            all_objects = [model.tokenizer.tokenize(x)[0] for x in all_objects]
    else:
        all_objects = None

    results_dict = {}

    if args.baseline:
        for prompt_id, prompt in enumerate(prompts):
            filtered_data, predictions = create_majority_baseline(data)
            results_dict[prompt] = {"data": filtered_data, "predictions": predictions}
    else:
        for prompt_id, prompt in enumerate(prompts):
            results_dict[prompt] = []
            filtered_data, predictions = run_query(model, data, prompt, all_objects, args.bs)
            results_dict[prompt] = {"data": filtered_data, "predictions": predictions}


    # Evaluate on LAMA
    lama_acc = evaluate_lama(prompts[0], results_dict)
    wandb.run.summary['lama_acc'] = lama_acc

    # Group Eval
    group_acc = group_score_lama_eval(results_dict)
    wandb.run.summary['lama_group_acc'] = group_acc

    lm_results = parse_lm_results(results_dict, all_objects)

    analyze_results(lm_results, patterns_graph)
    analyze_graph(patterns_graph)

    if 'models' in model_name or 'nyu' in model_name:
        model_name = model_name.replace('/', '_')
    pattern = args.patterns_file.split('/')[-1].split('.')[0]
    with open('data/output/predictions_lm/trex_lms_vocab/{}_{}.json'.format(pattern, model_name), 'w') as f:
        json.dump(lm_results, f)


if __name__ == '__main__':
    main()
