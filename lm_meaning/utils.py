import json
from typing import List, Dict


def read_jsonl_file(filename: str) -> List[Dict]:

    dataset = []
    with open(filename) as f:
        for line in f:
            loaded_example = json.loads(line)
            dataset.append(loaded_example)

    return dataset


def read_json_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def load_prompts(filename: str) -> List[str]:
    prompts = []
    with open(filename, 'r') as fin:
        for row in fin:
            l = json.loads(row)
            prompt = l['pattern']
            prompts.append(prompt)
    return prompts



