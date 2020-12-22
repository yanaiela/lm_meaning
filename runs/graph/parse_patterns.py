import argparse
from runs.ts_run import parallelize


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
]

relation2subj_obj = {
    'P101': {'subject': 'Turing', 'object': 'logic'},
    'P103': {'subject': 'Messmer', 'object': 'French'},
    'P1376': {'subject': 'Edmonton', 'object': 'Alberta'},
    'P1412': {'subject': 'Zohar', 'object': 'Hebrew'},
    'P159': {'subject': 'Maple', 'object': 'Waterloo'},
    'P276': {'subject': 'Tower', 'object': 'Paris'},
    'P30': {'subject': 'Glacier', 'object': 'Antarctica'},
    'P39': {'subject': 'Beaton', 'object': 'abbot'},
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
    'P527': {'subject': 'Book', 'object': 'chapters'},
    'P176': {'subject': 'Iphone', 'object': 'Apple'},
    'P178': {'subject': 'Iphone', 'object': 'Apple'},
    'P27': {'subject': 'John', 'object': 'England'},
    'P407': {'subject': 'Hamlet', 'object': 'English'},
    'P530': {'subject': 'France', 'object': 'Italy'},
    'P37': {'subject': 'Italy', 'object': 'Italian'},
    'P413': {'subject': 'John', 'object': 'keeper'},
    'P364': {'subject': 'Lost', 'object': 'English'},
    'P108': {'subject': 'Steve', 'object': 'Apple'},
    'P937': {'subject': 'Enki', 'object': 'Paris'},
    'P136': {'subject': 'Herbie', 'object': 'jazz'},
    'P1303': {'subject': 'Walter', 'object': 'saxophone'},
    'P17': {'subject': 'Simcoe', 'object': 'Canada'},
    'P127': {'subject': 'Atari', 'object': 'Atari'},
    'P31': {'subject': 'Preston', 'object': 'village'},
    'P361': {'subject': 'arithmetic', 'object': 'mathematics'},
    'P140': {'subject': 'Christianization', 'object': 'Christianity'},
    'P1001': {'subject': 'PM', 'object': 'Canada'},
    'P264': {'subject': 'Buddy', 'object': 'Brunswick'},
    'P279': {'subject': 'Champagne', 'object': 'wine'}
}


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("-patterns", "--patterns", type=str, help="patterns file",
                       default="runs/core/patterns.txt")
    args = parse.parse_args()

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
