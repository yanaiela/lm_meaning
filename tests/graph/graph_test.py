import argparse

from lm_meaning.utils import read_graph
from runs.utils import get_lama_patterns


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--graph_file", type=str, help="Path to templates for each prompt",
                       default="data/pattern_data/graphs/")
    parse.add_argument("-lama_patterns", "--lama_patterns", type=str, help="patterns file",
                       default="data/trex/data/relations.jsonl")

    args = parse.parse_args()

    for lama_pattern in get_lama_patterns(args.lama_patterns):
        print(lama_pattern, end=', ')
        try:
            patterns_graph = read_graph(args.graph_file + lama_pattern + '.graph')
        except:
            continue

        transitivity_issues = []
        for node in patterns_graph.nodes:
            for ent_node in patterns_graph.successors(node):
                for ent_ent_node in patterns_graph.successors(ent_node):
                    if node == ent_ent_node:
                        continue
                    if [node, ent_node] in patterns_graph.edges and [ent_node, ent_ent_node] in patterns_graph.edges:
                        if [node, ent_ent_node] not in patterns_graph.edges:
                            transitivity_issues.append([node.lm_pattern + f' ({node.extended_lemma})',
                                                        ent_node.lm_pattern + f' ({ent_node.extended_lemma})',
                                                        ent_ent_node.lm_pattern + f' ({ent_ent_node.extended_lemma})'])
        if len(transitivity_issues) > 0:
            with open('logs/{}.txt'.format(lama_pattern), 'w') as f:
                for line in transitivity_issues:
                    f.write(' --> '.join(line) + '\n')
        print()
        # break
    # print('success')


if __name__ == '__main__':
    main()

