import argparse
from memorization_runs.ts_run import parallelize
from memorization_runs.utils import get_lama_patterns, get_servers


nodes = get_servers()


# ┌──────────┐
# │ encoders │
# └──────────┘
encoders = [
            'bert-base-cased',
            'bert-large-cased',
            'google/multiberts-seed_*',
            'albert-base-v1',
            'albert-large-v1',
            'albert-xlarge-v1',
            'albert-xxlarge-v1',
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
    parse.add_argument("-random_weights", "--random_weights", type=str, help="use random weights",
                       default="false")
    args = parse.parse_args()

    relations = get_lama_patterns(args.patterns)

    if args.random_weights == 'false':
        out_sub_dir = 'bert_lama_unpatterns'
    else:
        out_sub_dir = 'randw_bert_lama_unpatterns'

    cartesian_product = []
    for encoder_inner in encoders:
        if '*' in encoder_inner:
            for i in range(25):
                encoder = encoder_inner.replace('*', str(i))
                for relation_id in relations:
                    cartesian_product.append([f'memorization_data/trex_lms_vocab/{relation_id}.jsonl',
                                              encoder,
                                              f'data/unpattern_data/{relation_id}.jsonl',
                                              f'memorization_data/output/predictions_lm/{out_sub_dir}/{relation_id}_{encoder}.json',
                                              args.random_weights
                                              ])
        else:
            encoder = encoder_inner
            for relation_id in relations:
                cartesian_product.append([f'memorization_data/trex_lms_vocab/{relation_id}.jsonl',
                                          encoder,
                                          f'data/unpattern_data/{relation_id}.jsonl',
                                          f'memorization_data/output/predictions_lm/{out_sub_dir}/{relation_id}_{encoder}.json',
                                          args.random_weights
                                          ])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/memorization_runs/encode/encode.sh',
                on_gpu=True, dry_run=args.dry_run)
