import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = ['bert-base-cased',
            'bert-large-cased',
            'roberta-base',
            'roberta-large',
            ]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="runs/core/patterns.txt")
    args = parse.parse_args()

    with open(args.patterns, 'r') as f:
        relations = f.readlines()
        relations = [x.strip() for x in relations]

    cartesian_product = []
    for relation_id in relations:
        for lm in encoders:
            cartesian_product.append([f'data/output/unpatterns/{relation_id}_{lm}.jsonl',
                                      f'data/unpattern_data/{relation_id}.jsonl'])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/evaluate/run_unpattern_eval.sh',
                on_gpu=True, dry_run=args.dry_run)
