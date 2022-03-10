import argparse
from memorization_runs.ts_rerun import parallelize
import wandb
from datetime import datetime
import json


# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [

    # 'nlp01',
    # 'nlp02',
    # 'nlp03',
    # 'nlp04',
    # 'nlp05',
    # 'nlp06',
    'nlp07',
    'nlp08',
    'nlp09',
    'nlp10',
    # 'nlp11',
    'nlp12',
    # 'nlp13',
    'nlp14',
    'nlp15',
    'nlp16',
    'nlp17',
]


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    args = parse.parse_args()

    api = wandb.Api()

    start_time = datetime.strptime("2022-03-10T00:00:00", '%Y-%m-%dT%H:%M:%S')

    runs = api.runs("consistency/memorization")
    error_runs = []

    for run in runs:
        date_time_obj = datetime.strptime(run.created_at, '%Y-%m-%dT%H:%M:%S')
        if date_time_obj > start_time and run.state == 'failed':
            meta = json.load(run.file("wandb-metadata.json").download())
            program = ["/home/nlp/lazary/anaconda3/envs/memorization/bin/python"] + [meta["program"]] + meta["args"]
            error_runs.append(program)

    cartesian_product = []

    parallelize(nodes, cartesian_product, on_gpu=True, dry_run=args.dry_run)
