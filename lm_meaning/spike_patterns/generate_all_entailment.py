import argparse
from collections import defaultdict

import pandas as pd


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-patterns_file", "--patterns_file", type=str, help=" path to the pattern file",
                       default="data/pattern_data/P449.tsv")
    parse.add_argument("-lemmas_file", "--lemmas_file", type=str, help=" path to the lemmas file",
                       default="data/pattern_data/P449_entailment_lemmas.tsv")
    parse.add_argument("-output_file", "--output_file", type=str, help="output file",
                       default="data/pattern_data/entailed_lemmas_extended/P449_entailment_lemmas.tsv")

    args = parse.parse_args()

    df_patterns = pd.read_csv(args.patterns_file, sep="\t")

    with open(args.lemmas_file, "r") as f:
        lines = f.readlines()

    asymetric_lemmas = defaultdict(list)
    all_lemmas = set([l.strip().split("\t")[0] for l in lines])
    print(all_lemmas)

    for line in lines[:]:
        # print(line)
        l, other_lemmas = line.strip().split("\t")
        if "*" in other_lemmas:
            mode = "not-entailed"
        elif "+" in other_lemmas:
            mode = "entailed"

        if mode == "not-entailed":
            not_entailed = other_lemmas.split("*/")[1].split(",")
        elif mode == "entailed":
            entailed = other_lemmas.split("+/")[1].split(",")
            not_entailed = [l2 for l2 in all_lemmas if l2 not in entailed and l2 != l]

        print("Not entailed from lemma {} are: {}".format(l, not_entailed))
        asymetric_lemmas[l].extend(not_entailed)

    all_lemmas = list(set(df_patterns["EXTENDED-LEMMA"].tolist()))
    with open(args.output_file, "w") as f:

        f.write("LEMMA\tNOT-ENTAILED\n")
        for lemma in all_lemmas:

            if lemma not in asymetric_lemmas.keys():
                not_entailed = list(set(asymetric_lemmas.keys()))
            else:
                not_entailed = list(set(asymetric_lemmas[lemma]))

            f.write(lemma + "\t" + ",".join(not_entailed) + "\n")


if __name__ == '__main__':
    main()
