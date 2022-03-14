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
            'google_multiberts-seed_*',
            ]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file, in case using all patterns, use the"
                                                                 "'all' argument",
                       default="data/trex/data/relations.jsonl")
    parse.add_argument("-random_weights", "--random_weights", type=str, help="use random weights",
                       default="false")
    args = parse.parse_args()

    cartesian_product = []

    if args.patterns == 'all':
        for encoder in encoders:
            if "*" in encoder:
                for i in range(25):
                    encoder_inner = encoder.replace('*', str(i))
                    cartesian_product.append([encoder_inner, 'all', args.random_weights])
            else:
                cartesian_product.append([encoder, 'all', args.random_weights])
    else:
        relations = get_lama_patterns(args.patterns)
        for rel in relations:
            for encoder in encoders:
                if "*" in encoder:
                    for i in range(25):
                        encoder_inner = encoder.replace('*', str(i))
                        cartesian_product.append([encoder_inner, rel, args.random_weights])
                else:
                    cartesian_product.append([encoder, rel, args.random_weights])

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
