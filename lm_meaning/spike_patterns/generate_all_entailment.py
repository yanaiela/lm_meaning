import sys
import pandas as pd
from collections import defaultdict

patterns_file = sys.argv[1]
lemmas_file = sys.argv[2]

df_patterns = pd.read_csv(patterns_file, sep = "\t")

with open(lemmas_file, "r") as f:
	lines = f.readlines()
	
asymetric_lemmas = defaultdict(list)
for line in lines[:]:
	#print(line)
	l, not_entailed = line.strip().split("\t")
	not_entailed = not_entailed.split("*/")[1].split(",")
	print("Not entailed from lemma {} are: {}".format(l, not_entailed))
	for not_entailed_l in not_entailed:
	
		asymetric_lemmas[l].extend(not_entailed)
		

all_lemmas = df_patterns["EXTENDED-LEMMA"].tolist()
with open(lemmas_file.split(".tsv")[0]+".extended.tsv", "w") as f:

	f.write("LEMMA\tNOT-ENTAILED\n")
	for lemma in all_lemmas:

		if lemma not in asymetric_lemmas.keys():
			not_entailed = list(asymetric_lemmas.keys())
		else:
			not_entailed = asymetric_lemmas[lemma]

		f.write(lemma + "\t" + ",".join(not_entailed) + "\n")
