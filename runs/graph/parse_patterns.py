import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
]

relation2subj_obj = {
    'P495': {'subject': 'golf', 'object': 'Scotland'},
    'P740': {'subject': 'Nikon', 'object': 'Tokyo'},
    'P36': {'subject': 'France', 'object': 'Paris'},
    'P19': {'subject': 'John', 'object': 'England'},
    'P20': {'subject': 'John', 'object': 'England'},
    'P106': {'subject': 'John', 'object': 'lawyer'},
    'P131': {'subject': 'John', 'object': 'Spain'},
    'P190': {'subject': 'Doha', 'object': 'Ankara'},
    'P499': {'subject': 'ABC', 'object': 'Lost'},
    'P138': {'subject': 'The rainforest frog Eleutherodactylus pecki', 'object': 'Robert M. Peck'},
    'P190': {'subject': 'Hanamaki, Japan', 'object': 'Sigmundsherberg, Austria'},
    'P47': {'subject': 'Israel', 'object': 'Syria'},
    'P102': {'subject': 'Trump', 'object': 'Republican'},
    'P527': {'subject': 'Book', 'object': 'chapters'},
}


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="runs/core/patterns.txt")
    args = parse.parse_args()

    with open(args.patterns, 'r') as f:
        relations = f.readlines()
        relations = [x.strip() for x in relations]

    cartesian_product = []
    for relation_id in relations:
        subj = relation2subj_obj[relation_id]['subject']
        obj = relation2subj_obj[relation_id]['object']
        cartesian_product.append([f'data/pattern_data/{relation_id}.tsv',
                                  subj,
                                  obj])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/graph/parse_patterns.sh',
                on_gpu=False, dry_run=args.dry_run)
