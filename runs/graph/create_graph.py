import argparse
from runs.ts_run import parallelize
from runs.utils import get_lama_patterns


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
]


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
        cartesian_product.append([f'data/pattern_data/parsed/{relation_id}.jsonl',
                                  f'data/pattern_data/entailed_lemmas_extended/{relation_id}.tsv',
                                  f'data/pattern_data/graphs_tense/{relation_id}.graph'])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/graph/create_graph.sh',
                on_gpu=False, dry_run=args.dry_run)
