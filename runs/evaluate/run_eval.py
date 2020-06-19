import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp15',
]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘

runs_dic = {
    'number': {
        'task': 'Numeric',
        'file': 's3://lminstructions/instructions/number2int.jsonl.gz',
        'split': 'dev',
        'encoders': [
            'bert-base-uncased',
            'bert-large-uncased',
            'roberta-base',
            'roberta-large']

    },

}

if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    args = parse.parse_args()

    cartesian_product = []
    for data_type, vals in runs_dic.items():
        task = vals['task']
        file = vals['file']
        split = vals['split']
        for encoder in vals['encoders']:
            cartesian_product.append([task, file, split, encoder])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/evaluate/run_eval.sh',
                on_gpu=True, dry_run=args.dry_run)
