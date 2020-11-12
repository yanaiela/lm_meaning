import sys
import pandas as pd
from collections import defaultdict

patterns_file = sys.argv[1]
lemmas_file = sys.argv[2]

df_patterns = pd.read_csv(patterns_file, sep = "\t")

with open(lemmas_file, "r") as f:
	lines = f.readlines()
	
asymetric_lemmas = defaultdict(list)
all_lemmas = set([l.strip().split("\t")[0] for l in lines])
print(all_lemmas)

for line in lines[:]:
	#print(line)
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
		

all_lemmas = lits(set(df_patterns["EXTENDED-LEMMA"].tolist()))
with open(lemmas_file.split(".tsv")[0]+".extended.tsv", "w") as f:

	f.write("LEMMA\tNOT-ENTAILED\n")
	for lemma in all_lemmas:

		if lemma not in asymetric_lemmas.keys():
			not_entailed = list(set(asymetric_lemmas.keys()))
		else:
			not_entailed = list(set(asymetric_lemmas[lemma]))

		f.write(lemma + "\t" + ",".join(not_entailed) + "\n")
