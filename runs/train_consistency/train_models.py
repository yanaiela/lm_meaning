import argparse

from runs.ts_run import parallelize

# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp09',
    'nlp10',
    'nlp12',
    'nlp15',
]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    args = parse.parse_args()

    cartesian_product = []
    for n_graphs in [3, 6, 11, 21]:
        for n_patterns in [100, 200, 500, 1000, 2000]:

            cartesian_product.append([f'data/enailment_train/consistancy_relation_{n_graphs}_{n_patterns}/train.txt',
                                      f'models/consistency/bert_base_cased/{n_graphs}_{n_patterns}/',
                                      ])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/train_consistency/train_models.sh',
                on_gpu=True, dry_run=args.dry_run)
