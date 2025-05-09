import networkx as nx
from networkx.readwrite import json_graph

from pararel.patterns.graph_types import PatternNode, EdgeType


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

    def test_encode_decode(self):
        pattern1 = PatternNode("[X] was aired on [Y].", "<>subject:Lost $was $aired $on object:[w={}]ABC.",
                               'air', 'is-air-on', 'past', "[X]:The Crown :was :aired on [Y]:Netflix")
        pattern2 = PatternNode("[X] was released on [Y].", "<>subject:Lost $was $released $on object:[w={}]ABC.",
                               'release', 'is-release-on', 'past')

        # generating a graph
        DiG = nx.DiGraph()
        DiG.add_node(pattern1)
        DiG.add_node(pattern2)
        DiG.add_edge(pattern1, pattern2, edge_type=EdgeType.syntactic)

        # graph_data is of json data, which can be dumped to a file
        graph_data = json_graph.node_link_data(DiG)
        DiG_reconstruct = json_graph.node_link_graph(graph_data)

        assert nx.is_isomorphic(DiG, DiG_reconstruct)
