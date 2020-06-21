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
encoders = ['bert-base-uncased',
            'bert-large-uncased',
            'roberta-base',
            'roberta-large',
            'albert-base-v2',
            'albert-large-v2',
            'albert-xlarge-v2',
            'albert-xxlarge-v2',
            'xlm-roberta-base',
            'xlm-roberta-large',
            ]

# ┌────────┐
# │ splits │
# └────────┘
splits = ['dev']

# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘

runs_dic = {
    'number': {
        'task': 'Numeric',
        'file': 's3://lminstructions/instructions/number2int.jsonl.gz',
        'split': 'dev',
    },
    'plural': {
        'task': 'Plural',
        'file': 's3://lminstructions/instructions/plurals.jsonl.gz',
        'split': 'dev',
    },
    'past': {
        'task': 'Past',
        'file': 's3://lminstructions/instructions/past.jsonl.gz',
        'split': 'dev',
    },
    'truncate': {
        'task': 'Truncate',
        'file': 's3://lminstructions/instructions/truncate.jsonl.gz',
        'split': 'dev',
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
        for split in splits:
            for encoder in encoders:
                cartesian_product.append([task, file, split, encoder])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/evaluate/run_eval.sh',
                on_gpu=True, dry_run=args.dry_run)
