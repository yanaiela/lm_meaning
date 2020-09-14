import json

from spike.search.data_set_connections import get_data_sets_connections
from pathlib import Path


def get_spike_objects(config_path='./my_config.yaml'):
    data_sets_connections = get_data_sets_connections(Path(config_path))
    engine = data_sets_connections.of("wiki").engine
    annotator = data_sets_connections.of("wiki").annotator
    return engine, annotator


def get_relations_data(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()

    lines = [json.loads(x) for x in lines]
    return lines


def get_patterns(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]

    return lines


def dump_json(data, out_file):
    with open(out_file, 'w') as f:
        json.dump(data, f)
