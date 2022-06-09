import argparse
from memorization_runs.ts_rerun import parallelize
from memorization_runs.utils import get_servers
import wandb
from datetime import datetime
import json
from tqdm import tqdm


nodes = get_servers()


# ┌──────────────────────┐
# │ generate experiments │
# └──────────────────────┘


if __name__ == '__main__':

    parse = argparse.ArgumentParser("")
    parse.add_argument("-dry_run", "--dry_run", type=bool, help="flag to only print commands and not execute them",
                       default=False)
    parse.add_argument("--no_gpu", action='store_false')
    args = parse.parse_args()

    api = wandb.Api()

    start_time = datetime.strptime("2022-03-14T00:00:00", '%Y-%m-%dT%H:%M:%S')

    runs = api.runs("consistency/memorization", filters={'state': 'failed'})
    error_runs = []

    for run in tqdm(runs):
        date_time_obj = datetime.strptime(run.created_at, '%Y-%m-%dT%H:%M:%S')
        if date_time_obj > start_time and run.state == 'failed':
            meta = json.load(run.file("wandb-metadata.json").download(replace=True))
            program = ["/home/nlp/lazary/anaconda3/envs/memorization/bin/python"] + [meta["program"]] + meta["args"]
            error_runs.append(program)

    parallelize(nodes, error_runs, on_gpu=args.no_gpu, dry_run=args.dry_run)
