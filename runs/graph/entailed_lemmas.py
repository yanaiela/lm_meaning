import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
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
        cartesian_product.append([f'data/pattern_data/{relation_id}.tsv',
                                  f'data/pattern_data/{relation_id}_entailment_lemmas.tsv',
                                  f'data/pattern_data/entailed_lemmas_extended/{relation_id}_entailment_lemmas.tsv'])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/graph/entailed_lemmas.sh',
                on_gpu=False, dry_run=args.dry_run)
