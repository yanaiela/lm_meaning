import argparse
from memorization_runs.ts_run import parallelize
from memorization_runs.utils import get_lama_patterns


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    #'nlp15',
    # 'nlp01',
    'nlp02',
    # 'nlp03',
    # 'nlp04',
    'nlp05',
    'nlp06',
    'nlp07',
    # 'nlp08',
    'nlp09',
    'nlp10',
    'nlp11',
    'nlp12',
    'nlp13',
    # 'nlp14',
    'nlp16',
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
            ]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="data/trex/data/relations.jsonl")
    args = parse.parse_args()

    relations = get_lama_patterns(args.patterns)

    cartesian_product = []
    for encoder in encoders:
        for relation_id in relations:
            cartesian_product.append([f'memorization_data/trex_lms_vocab/{relation_id}.jsonl',
                                      encoder,
                                      f'data/unpattern_data/{relation_id}.jsonl',
                                      f'memorization_data/output/predictions_lm/bert_lama_unpatterns/{relation_id}_{encoder}.json'
                                      ])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/memorization_runs/encode/encode.sh',
                on_gpu=True, dry_run=args.dry_run)
