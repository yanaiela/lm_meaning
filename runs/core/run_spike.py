import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    'nlp02',
]


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
        cartesian_product.append([f'data/trex/data/TREx/{relation_id}.jsonl',
                                  f'data/pattern_data/parsed/{relation_id}.jsonl',
                                  f'data/output/spike_results/paraphrases/{relation_id}.json'
                                  ])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/core/run_spike.sh',
                on_gpu=False, dry_run=args.dry_run)
