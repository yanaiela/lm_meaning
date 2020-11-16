import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp10',
    'nlp11',
    'nlp12',
    'nlp13',
    'nlp14',
    'nlp15',
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = ['bert-base-cased',
            'bert-large-cased',
            #'roberta-base',
            #'roberta-large',
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
        for encoder in encoders:
            cartesian_product.append([f'data/unpattern_data/{relation_id}.jsonl',
                                      f'data/trex/data/TREx/{relation_id}.jsonl',
                                      encoder,
                                      f'data/output/unpatterns/{relation_id}_{encoder}.jsonl'])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/core/run_lm.sh',
                on_gpu=True, dry_run=args.dry_run)
