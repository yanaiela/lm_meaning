import argparse
from memorization_runs.ts_run import parallelize
from memorization_runs.utils import get_lama_patterns


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    'nlp02',
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
    'nlp16',
    'nlp17',
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = [
            'bert-base-cased',
            'bert-large-cased',
            'nyu-mll/roberta-base-1B-1',
            'nyu-mll/roberta-base-100M-1',
            'nyu-mll/roberta-base-10M-1',
            'nyu-mll/roberta-med-small-1M-1',
            'google/multiberts-seed_*',
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
    for encoder in encoders:
        if "*" in encoder:
            for i in range(25):
                encoder_inner = encoder.replace('*', str(i))
                cartesian_product.append([encoder_inner])
        else:
            cartesian_product.append([encoder])

    # Subject-object cooccurrences
    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/memorization_runs/causality/causal_subj_obj_cooccurrence.sh',
                on_gpu=False, dry_run=args.dry_run)

    # Pattern-object cooccurrences
    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/memorization_runs/causality/causal_pat_obj_cooccurrence.sh',
                on_gpu=False, dry_run=args.dry_run)

    # Memorization
    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/memorization_runs/causality/causal_memorization.sh',
                on_gpu=False, dry_run=args.dry_run)
