import argparse
import pandas as pd
from runs.utils import get_lama_patterns


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--patterns_file", type=str, help="Path to templates for each prompt",
                       default="data/pattern_data/")
    parse.add_argument("--lemmas_file", type=str, help="lemmas file",
                       default="data/pattern_data/")
    parse.add_argument("-lama_patterns", "--lama_patterns", type=str, help="patterns file",
                       default="data/trex/data/relations.jsonl")

    args = parse.parse_args()

    for lama_pattern in get_lama_patterns(args.lama_patterns):
        print(lama_pattern, end=', ')

        df_patterns = pd.read_csv(args.patterns_file + lama_pattern + '.tsv', sep='\t')
        df_lemmas = pd.read_csv(args.lemmas_file + lama_pattern + '_entailment_lemmas.tsv',
                                header=0, names=['lemma', 'followed'], sep='\t')

        extended_lemmas = df_patterns['EXTENDED-LEMMA'].unique()

        entailing_lemmas = df_lemmas['lemma'].tolist()
        entailed_lemmas = df_lemmas['followed'].tolist()

        patterns = df_patterns['RULE'].tolist()
        for p in patterns:
            if patterns.count(p) != 1:
                print('double pattern:', p)

        for lemma in entailing_lemmas:
            lemma_strip = lemma.strip()
            if lemma_strip not in extended_lemmas:
                print('err first column', lemma_strip, end=', ')

        for row in entailed_lemmas:
            for lemma in row.split('/')[1].strip().split(','):
                lemma_strip = lemma.strip()
                if lemma_strip == '':
                    continue
                if lemma_strip not in extended_lemmas:
                    print('err second column, ', lemma_strip, end=', ')

        print()
    # print('success')


if __name__ == '__main__':
    main()

