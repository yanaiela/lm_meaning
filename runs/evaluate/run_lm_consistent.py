import argparse
from runs.ts_run import parallelize
from runs.utils import get_lama_patterns


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp15',
    'nlp01',
    'nlp02',
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
]


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = [
            'bert-base-cased',
            'roberta-base',
            'albert-base-v2',
            'bert-large-cased',
            'bert-large-cased-whole-word-masking',
            'roberta-large',
            'albert-xxlarge-v2',

            'nyu-mll/roberta-base-1B-1',
            'nyu-mll/roberta-base-100M-1',
            'nyu-mll/roberta-base-10M-1',
            'nyu-mll/roberta-med-small-1M-1',

            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_no-typed_no-wiki_lama_0_0.1_5/checkpoint-6/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_no-typed_no-wiki_lama_0_0.1_5/checkpoint-12/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_no-typed_no-wiki_lama_0_0.1_5/checkpoint-18/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_no-typed_no-wiki_lama_0_0.1_5/checkpoint-24/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_no-typed_no-wiki_lama_0_0.1_5/checkpoint-30/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_no-typed_no-wiki_lama_0_0.1_5/checkpoint-36/',

            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-6/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-12/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-18/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-24/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-30/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-36/',

            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_no-lama_0_0.1_0/checkpoint-6/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_no-lama_0_0.1_0/checkpoint-12/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_no-lama_0_0.1_0/checkpoint-18/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_no-lama_0_0.1_0/checkpoint-24/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_no-lama_0_0.1_0/checkpoint-30/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_dkl_typed_no-wiki_no-lama_0_0.1_0/checkpoint-36/',

            'models/nora/consistency_global_100_3_P138-P449-P37_bert-large-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-6/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-large-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-12/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-large-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-18/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-large-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-24/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-large-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-30/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-large-cased_dkl_typed_no-wiki_lama_0_0.1_5/checkpoint-36/',

            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_no_no-typed_no-wiki_lama_0_0.1_5/checkpoint-6/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_no_no-typed_no-wiki_lama_0_0.1_5/checkpoint-12/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_no_no-typed_no-wiki_lama_0_0.1_5/checkpoint-18/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_no_no-typed_no-wiki_lama_0_0.1_5/checkpoint-24/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_no_no-typed_no-wiki_lama_0_0.1_5/checkpoint-30/',
            'models/nora/consistency_global_100_3_P138-P449-P37_bert-base-cased_no_no-typed_no-wiki_lama_0_0.1_5/checkpoint-36/',

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
            cartesian_product.append([f'data/pattern_data/parsed/{relation_id}.jsonl',
                                      f'data/trex_lms_vocab/{relation_id}.jsonl',
                                      #f'data/eval_100_left/{relation_id}.jsonl',
                                      encoder,
                                      f'data/pattern_data/graphs_tense/{relation_id}.graph'
                                      ])

    parallelize(nodes, cartesian_product, '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/evaluate/run_lm_consistent.sh',
                on_gpu=True, dry_run=args.dry_run)
