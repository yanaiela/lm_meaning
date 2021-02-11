import argparse
import glob
import pickle
import random
import time
import os

from pararel.consistency import utils


def generate_data(num_relations, num_tuples, LAMA_path):

    num_relations -= 1
    graph_path = "data/pattern_data/graphs_sub/"
    relations_path =  glob.glob(graph_path + "*.graph")
    output_path = "data/enailment_train/global_consistancy_relation_"

    output_path = output_path + str(num_relations+1) + "/"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_path + timestr + "/"
    os.mkdir(output_path)

    output_path_true =  output_path + "train_consistancy_"
    output_path_mlm =  output_path + "train_mlm.txt"
    output_path_log = output_path + "log.txt"

    f_log = open(output_path_log, "w")
    f_mlm = open(output_path_mlm, "w")

    random.shuffle(relations_path)


    f_log.write(str(num_tuples))
    f_log.write("\n")

    for i_relations, relation_path in enumerate(relations_path):

        with open(relation_path, "rb") as f:
            graph = pickle.load(f)
        relation = relation_path.split("/")[-1].split(".")[0]

        
        f_true = open(output_path_true + relation + ".txt", "w")
        #f_mlm = open(output_path_mlm + relation + ".txt", "w")


        f_log.write(relation)
        f_log.write("\n")
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
                break
        f_true.close()
        #f_mlm.close()
        if i_relations==num_relations:
            break
    f_mlm.close()
    f_log.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_relations', '-nr', type=int, default=5, help='number of relations')
    parser.add_argument('--num_tuples', '-nt', type=int, default=2000, help='number of tuples')
    parser.add_argument('--LAMA_path', '-lama', type=str, default="/mounts/data/proj/kassner/LAMA/data/TREx/", help='number of tuples')


    args = parser.parse_args()

    generate_data(args.num_relations, args.num_tuples, args.LAMA_path)

if __name__ == "__main__":
    main()

