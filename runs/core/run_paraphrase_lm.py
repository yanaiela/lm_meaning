import argparse
from runs.ts_run import parallelize
from runs.utils import get_lama_patterns


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp04',
    'nlp05',
    'nlp07',
    'nlp09',
    'nlp10',
    # 'nlp11',
    'nlp12',
    'nlp13',
    'nlp14',
    'nlp15',
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = ['bert-base-cased',
            'bert-large-cased',
            'bert-large-cased-whole-word-masking',
            'roberta-base',
            'roberta-large',
            'albert-base-v2',
            'albert-xxlarge-v2',
            
            'nyu-mll/roberta-base-1B-1',
            'nyu-mll/roberta-base-100M-1',
            'nyu-mll/roberta-base-10M-1',
            'nyu-mll/roberta-med-small-1M-1'
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
    for relation_id in relations:
        for encoder in encoders:

            prediction_encoder_name = encoder
            if 'nyu-mll' in prediction_encoder_name:
                prediction_encoder_name = encoder.split('/')[-1]
            cartesian_product.append([f'data/pattern_data/parsed/{relation_id}.jsonl',
                                      f'data/trex_lms_vocab/{relation_id}.jsonl',
                                      encoder,
                                      f'data/output/predictions_lm/trex_lms_vocab/{relation_id}_{prediction_encoder_name}.json'])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/core/run_lm.sh',
                on_gpu=True, dry_run=args.dry_run)
