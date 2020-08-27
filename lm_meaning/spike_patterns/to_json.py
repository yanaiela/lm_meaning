import json
import sys


fname = sys.argv[1]
with open(fname, "r") as f:
    lines = f.readlines()
    

print(fname)
with open(fname.replace(".tsv","")+".jsonl", "w") as f:
    for line in lines[1:]:

       vals = line.strip().split("\t")
       vals.pop(-2) # remove old lemma/syntax - not relevant
       tense,lemma = vals[-1].split(",")
       pattern = vals[0]
       example = vals[1]
       vals = vals[:-1] + vals[-1].split(",")
       
       # add spike syntax
       spike_query = pattern.replace("X", "<>X:[w]").replace("Y", "<>Y:[w]")
       spike_query = spike_query.split(" ")
       
       for i,w in enumerate(spike_query):
        if "X" not in w and "Y" not in w:
            spike_query[i] = "$" + w
       spike_query =  " ".join(spike_query)
       
       # save as json
       
       vals = [pattern, example, tense, lemma, spike_query]
       keys = ["pattern", "example", "tense", "lemma", "spike_query"]
       dict = {k:v for k,v in zip(keys, vals)}

       f.write(json.dumps(dict) + "\n")
