import argparse
import json


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-patterns_file", "--patterns_file", type=str, help=" path to the pattern file",
                       default="data/pattern_data/P449.tsv")
    parse.add_argument("-subject", "--subject", type=str, help="name of the subject",
                       default="Friends")
    parse.add_argument("-object", "--object", type=str, help="name of the object",
                       default="NBC")
    args = parse.parse_args()

    fname = args.patterns_file
    subj, obj = args.subject, args.object

    with open(fname, "r") as f:
        lines = f.readlines()

    print(fname)
    with open(fname.replace(".tsv", "") + ".jsonl", "w") as f:
        for line in lines[1:]:

            vals = line.strip().split("\t")

            pattern = vals[0]
            example = vals[1]

            lemma, extended_lemma, tense = vals[2:]
            # vals = vals[:-1] + vals[-1].split(",")

            # add spike syntax
            spike_query = pattern.replace("[X]", "<>subject:" + subj).replace("[Y]", "object:[w={}]" + obj)
            spike_query = spike_query.split(" ")

            for i, w in enumerate(spike_query):
                if "subject" not in w and "object" not in w:
                    spike_query[i] = "$" + w
            spike_query = " ".join(spike_query)

            # save as json
            vals = [pattern, example, lemma, extended_lemma, tense, spike_query]
            keys = ["pattern", "example", "lemma", "extended_lemma", "tense", "spike_query"]
            dict = {k: v for k, v in zip(keys, vals)}

            f.write(json.dumps(dict) + "\n")


if __name__ == '__main__':
    main()
