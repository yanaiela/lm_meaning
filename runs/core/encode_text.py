import argparse
from runs.ts_run import parallelize
from runs.utils import get_lama_patterns

# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    #'nlp02',
    'nlp03',
    #'nlp04',
    'nlp05',
    'nlp06',
    'nlp07',
    'nlp08',
    #'nlp09',
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
            'nyu-mll/roberta-med-small-1M-1',

            #'models/nora/consistency_global_100_3_P138-P37-P449_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-12/',
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
        if relation_id != 'P106': continue
        for encoder in encoders:

            prediction_encoder_name = encoder
            if 'nyu-mll' in prediction_encoder_name:
                prediction_encoder_name = encoder.split('/')[-1]
            if 'nora' in prediction_encoder_name:
                prediction_encoder_name = 'consistency_global_100_3_P138-P37-P449_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5_checkpoint-12'
            cartesian_product.append([f'data/pattern_data/parsed/{relation_id}.jsonl',
                                      f'data/trex_lms_vocab/{relation_id}.jsonl',
                                      encoder,
                                      f'data/output/representations/{relation_id}_{prediction_encoder_name}.npy'])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/core/encode_text.sh',
                on_gpu=False, dry_run=args.dry_run)
