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



def generate_data(num_relations, num_tuples, relations_given, LAMA_path):

    #num_relations -= 1
    graph_path = "data/pattern_data/graphs_tense/"
    relations_path =  glob.glob(graph_path + "*.graph")
    output_path = "data/consistency/consistency_global_"

    random.shuffle(relations_path)
    relation_path_keep = []
    metadata = "_"
    if relations_given!="":
        for relation_path in relations_path:
            relation = relation_path.split("/")[-1].split(".")[0]
            if relation in relations_given.split(","):
                relation_path_keep.append(relation_path)
                metadata+= relation
                metadata+="-"
    if len(relation_path_keep) < num_relations:
        for relation_path in relations_path:
            if relation_path not in relation_path_keep:
                relation = relation_path.split("/")[-1].split(".")[0]
                relation_path_keep.append(relation_path)
                metadata+= relation
                metadata+="-"
                if len(relation_path_keep)==num_relations:
                    break
    metadata = metadata.strip("-")
    output_path = output_path + str(num_tuples) + "_" + str(num_relations) + metadata + "/"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path_true =  output_path + "train_"
    output_path_mlm =  output_path + "train_mlm.txt"
    f_mlm = open(output_path_mlm, "w")



    for relation_path in relation_path_keep:

        with open(relation_path, "rb") as f:
            graph = pickle.load(f)
        relation = relation_path.split("/")[-1].split(".")[0]

        f_true = open(output_path_true + relation + ".txt", "w")

        data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")
        random.shuffle(data)

        for i, d in enumerate(data):
            random.shuffle(data)
            for node in graph.nodes():
                pattern = node.lm_pattern

                pattern = pattern.replace("[X]", d["sub_label"])
                pattern = pattern.replace("[Y]", "[MASK]")
                pattern_mlm = pattern.replace("[MASK]", d["obj_label"])

                f_true.write(pattern)
                f_true.write("\n")
                f_mlm.write(pattern_mlm)
                f_mlm.write("\n")


            f_true.write("\n")

            if i >= num_tuples:
                print(i)
                break

        f_true.close()

    f_mlm.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations', '-nr', type=int, default=3, help='number of relations')
    parser.add_argument('--num_tuples', '-nt', type=int, default=100, help='number of tuples')
    parser.add_argument('--relations_given', '-r', type=str, default="", help='which relations')
    parser.add_argument('--LAMA_path', '-lama', type=str, default="/mounts/data/proj/kassner/lm_meaning/data/trex_lms_vocab/", help='number of tuples')


    args = parser.parse_args()

    generate_data(args.num_relations, args.num_tuples, args.relations_given, args.LAMA_path)

if __name__ == "__main__":
    main()

