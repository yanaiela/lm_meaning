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


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘

relations = [
    'P19',
    'P449',
]

if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    args = parse.parse_args()

    cartesian_product = []
    for relation_id in relations:
        cartesian_product.append([f'data/TREx_train/{relation_id}.jsonl',
                                  f'data/pattern_data/{relation_id}.jsonl',
                                  f'data/output/spike_results/TREx_train/{relation_id}.json'
                                  ])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/core/run_spike.sh',
                on_gpu=True, dry_run=args.dry_run)
