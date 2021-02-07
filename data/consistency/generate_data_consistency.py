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
 
    #num_relations = num_relations -1

    graph_path = "data/pattern_data/graphs_tense/"
    relations_path =  glob.glob(graph_path + "*.graph")
    output_path = "data/consistency/consistency_local_"

    random.shuffle(relations_path)
    relation_path_keep = []
    metadata = "_"
    if relations_given!="":
        for relation_path in relations_path:
            relation = relation_path.split("/")[-1].split(".")[0]
            if relation in relations_given.split(","):
                print(relation)
                relation_path_keep.append(relation_path)
                #metadata+= "_"
                metadata+= relation
                metadata+= "-"

    if len(relation_path_keep) < num_relations:
        for relation_path in relations_path:
            if relation_path not in relation_path_keep:
                relation = relation_path.split("/")[-1].split(".")[0]
                relation_path_keep.append(relation_path)
                #metadata+= "_" 
                metadata+= relation
                metadata+= "-"
                if len(relation_path_keep)==num_relations:
                    print(metadata)
                    break
    metadata = metadata.strip("-")
    output_path = output_path + str(num_tuples) + "_" + str(num_relations) + metadata + "/"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path_true =  output_path + "train.txt"
    output_path_mlm =  output_path + "train_mlm.txt"


    f_true = open(output_path_true, "w")
    f_mlm = open(output_path_mlm, "w")

    for i_relations, relation_path in enumerate(relation_path_keep):
        with open(relation_path, "rb") as f:
            graph = pickle.load(f)
        relation = relation_path.split("/")[-1].split(".")[0]
        # if relation not in  ["P138", "P37", "P449"]:
        """if relation not in  ["P31", "P1376", "P937"]:
            continue"""
        #f_log.write(relation)
        #f_log.write("\n")
        data = utils.read_jsonl_file(LAMA_path + relation + ".jsonl")
        random.shuffle(data)
        for edge in graph.edges():
            for i, d in enumerate(data):
                premise = edge[0].lm_pattern
                hypothesis = edge[1].lm_pattern

                premise = premise.replace("[X]", d["sub_label"])
                premise = premise.replace("[Y]", "[MASK]")
                premise_mlm = premise.replace("[MASK]", d["obj_label"])

                hypothesis = hypothesis.replace("[X]", d["sub_label"])
                hypothesis = hypothesis.replace("[Y]", "[MASK]")
                hyposthesis_mlm = hypothesis.replace("[MASK]", d["obj_label"])

                if i < num_tuples:
                    f_true.write(premise)
                    f_true.write("\n")
                    f_true.write(hypothesis)
                    f_true.write("\n")

                    f_mlm.write(premise_mlm)
                    f_mlm.write("\n")
                    f_mlm.write(hyposthesis_mlm)
                    f_mlm.write("\n")


                else:
                    continue

        """if i_relations==num_relations:
            break"""

    f_true.close()
    #f_log.close()
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

