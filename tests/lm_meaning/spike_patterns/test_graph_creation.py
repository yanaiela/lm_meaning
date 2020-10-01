import networkx as nx

from lm_meaning.spike_patterns.graph_types import PatternNode, EdgeType


class TestPluralInstruction:
    def test_build_challenge(self):

        pattern1 = PatternNode("[X] was aired on [Y].", "<>subject:Lost $was $aired $on object:[w={}]ABC.",
                               'air', 'is-air-on', 'past', "[X]:The Crown :was :aired on [Y]:Netflix")
        pattern2 = PatternNode("[X] was released on [Y].", "<>subject:Lost $was $released $on object:[w={}]ABC.",
                               'release', 'is-release-on', 'past')

        # generating a graph
        DiG = nx.DiGraph()
        DiG.add_node(pattern1)
        DiG.add_node(pattern2)
        DiG.add_edge(pattern1, pattern2, edge_type=EdgeType.syntactic)
        
        for node_in, node_out in DiG.out_edges(pattern1):
            assert node_in == pattern1
            assert node_out == pattern2
            assert DiG.edges[node_in, node_out]['edge_type'] == EdgeType.syntactic
