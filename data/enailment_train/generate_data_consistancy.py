import argparse
import glob
import pickle
import networkx as nx
import random
import time
import os

from itertools import permutations
from random import sample
from lm_meaning import utils



def generate_data(num_relations, num_tuples, LAMA_path):


    graph_path = "data/pattern_data/graphs/"
    relations_path =  glob.glob(graph_path + "*.graph")
    output_path = "data/enailment_train/consistancy_relation_"

    output_path = output_path + str(num_relations+1) + "/"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_path + timestr + "/"
    os.mkdir(output_path)

    output_path_true =  output_path + "train.txt"
    output_path_log = output_path + "log.txt"

    f_true = open(output_path_true, "w")
    f_log = open(output_path_log, "w")

    all_patterns = {}
    for relation_path in relations_path:

        with open(relation_path, "rb") as f:
             graph = pickle.load(f)
        if len(graph.nodes())==0:
            continue
        relation = relation_path.split("/")[-1].split(".")[0]
        all_patterns[relation] = {}
        all_patterns[relation]["patterns"] = []
        for node in graph.nodes():
             all_patterns[relation]["patterns"].append(node.lm_pattern)

        all_patterns[relation]["sub_obj"] = []
        data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")
        for d in data:
            all_patterns[relation]["sub_obj"].append([d["sub_label"], d["obj_label"]])

    random.shuffle(relations_path)

    f_log.write(str(num_tuples))
    f_log.write("\n")

    for i_relations, relation_path in enumerate(relations_path):
        with open(relation_path, "rb") as f:
            graph = pickle.load(f)
        relation = relation_path.split("/")[-1].split(".")[0]

        f_log.write(relation)
        f_log.write("\n")
        data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")
        random.shuffle(data)
        for edge in graph.edges():
            for i, d in enumerate(data):
                premise = edge[0].lm_pattern
                hypothesis = edge[1].lm_pattern

                premise = premise.replace("[X]", d["sub_label"])
                premise = premise.replace("[Y]", d["obj_label"])

                hypothesis = hypothesis.replace("[X]", d["sub_label"])
                hypothesis = hypothesis.replace("[Y]", d["obj_label"])
                if i < num_tuples:
                    f_true.write(premise)
                    f_true.write("\n")
                    f_true.write(hypothesis)
                    f_true.write("\n")

                else:
                    continue

        if i_relations==num_relations:
            break

    f_true.close()
    f_log.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations', '-nr', type=int, default=5, help='number of relations')
    parser.add_argument('--num_tuples', '-nt', type=int, default=100, help='number of tuples')
    parser.add_argument('--LAMA_path', '-lama', type=str, default="/mounts/data/proj/kassner/LAMA/data/TREx/", help='number of tuples')


    args = parser.parse_args()

    generate_data(args.num_relations, args.num_tuples, args.LAMA_path)

if __name__ == "__main__":
    main()

