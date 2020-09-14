import argparse
from runs.ts_run import parallelize
import json


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    'nlp03',
    'nlp04',
    'nlp05',
    'nlp06',
    'nlp07',
    'nlp08',
    'nlp09',
    'nlp10',
    'nlp11',
    'nlp12',
    'nlp13',
    'nlp14',
    'nlp15',
]


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="data/trex/data/relations.jsonl")
    args = parse.parse_args()

    with open(args.patterns, 'r') as f:
        relations = f.readlines()
        relations = [json.loads(x.strip())['relation'] for x in relations]

    cartesian_product = []
    for relation_id in relations:
        cartesian_product.append([relation_id,
                                  'data/trex/data/relations.jsonl',
                                  f'data/output/spike_results/preferences/{relation_id}.json'
                                  ])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/core'
                                          '/run_preference_spike.sh',
                on_gpu=False, dry_run=args.dry_run)
