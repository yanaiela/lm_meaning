import argparse
from runs.ts_run import parallelize
from runs.utils import get_lama_patterns


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    'nlp02',
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = ['bert-base-cased',
            'bert-large-cased',
            'bert-large-cased-whole-word-masking',
            'roberta-base',
            'roberta-large',
            'albert-base-v2',
            'albert-xxlarge-v2'
            ]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="data/trex/data/relations.jsonl")
    args = parse.parse_args()

    relations = get_lama_patterns(args.patterns)

    cartesian_product = []
    for relation_id in relations:
        for lm in encoders:
            cartesian_product.append([f'data/trex/data/TREx/{relation_id}.jsonl',
                                      f'data/output/predictions_lm/trex/{relation_id}_{lm}.json',
                                      f'data/lm_relations/{relation_id}.jsonl',
                                      f'data/pattern_data/{relation_id}.jsonl',
                                      f'data/pattern_data/graphs/{relation_id}.graph'])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/lm_entail/run_entail_eval.sh',
                on_gpu=True, dry_run=args.dry_run)
