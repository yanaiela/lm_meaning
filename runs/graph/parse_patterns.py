import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
]

relation2subj_obj = {
    'P101': {'subject': 'Alan Turing', 'object': 'logic'},
    'P103': {'subject': 'Pierre Messmer', 'object': 'French'},
    'P1376': {'subject': 'Edmonton', 'object': 'Alberta'},
    'P1412': {'subject': 'Itzik Zohar', 'object': 'Hebrew'},
    'P159': {'subject': 'Waterloo Maple', 'object': 'Waterloo'},
    'P276': {'subject': 'Eiffel Tower', 'object': 'Paris'},
    'P30': {'subject': 'Beardmore Glacier', 'object': 'Antarctica'},
    'P39': {'subject': 'David Beaton', 'object': 'abbot'},
    'P463': {'subject': 'Albania', 'object': 'NATO'},

    'P495': {'subject': 'golf', 'object': 'Scotland'},
    'P740': {'subject': 'Nikon', 'object': 'Tokyo'},
    'P36': {'subject': 'France', 'object': 'Paris'},
    'P19': {'subject': 'John', 'object': 'England'},
    'P20': {'subject': 'John', 'object': 'England'},
    'P106': {'subject': 'John', 'object': 'lawyer'},
    'P131': {'subject': 'John', 'object': 'Spain'},
    'P190': {'subject': 'Doha', 'object': 'Ankara'},
    'P449': {'subject': 'Lost', 'object': 'ABC'},
    'P138': {'subject': 'aristotelianism', 'object': 'Aristotle'},
    'P47': {'subject': 'Israel', 'object': 'Syria'},
    'P102': {'subject': 'Trump', 'object': 'Republican'},
    'P527': {'subject': 'Book', 'object': 'chapters'},
    'P166': {'subject': 'Obama', 'object': 'Nobel'},
    'P176': {'subject': 'Iphone', 'object': 'Apple'},
    'P178': {'subject': 'Iphone', 'object': 'Apple'},
    'P27': {'subject': 'John', 'object': 'England'},
    'P407': {'subject': 'Hamlet', 'object': 'English'},
    'P530': {'subject': 'France', 'object': 'Italy'},
    'P37': {'subject': 'Italy', 'object': 'Italian'},
    'P413': {'subject': 'John', 'object': 'keeper'},
    'P364': {'subject': 'Lost', 'object': 'English'},
}


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="runs/core/patterns.txt")
    args = parse.parse_args()

    # with open(args.patterns, 'r') as f:
    #     relations = f.readlines()
    #     relations = [x.strip() for x in relations]

    cartesian_product = []
    for relation_id, subj_obj in relation2subj_obj.items():
        subj = subj_obj['subject']
        obj = subj_obj['object']
        cartesian_product.append([f'data/pattern_data/{relation_id}.tsv',
                                  subj,
                                  obj,
                                  f'data/pattern_data/parsed/{relation_id}.jsonl'
                                  ])

    parallelize(nodes, cartesian_product,
                '/home/nlp/lazary/workspace/thesis/lm_meaning/runs/graph/parse_patterns.sh',
                on_gpu=False, dry_run=args.dry_run)
