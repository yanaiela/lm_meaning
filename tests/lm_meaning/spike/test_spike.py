from lm_meaning.spike.utils import equal_queries, get_spike_objects


class TestPluralInstruction:
    def test_build_challenge(self):
        _, spike_annotator = get_spike_objects()

        q1 = "<>subject:Lost $was $aired $on object:[w={}]ABC"
        q2 = "<>subject:Lost $was $premiered $on object:[w={}]ABC"
        q3 = "object:[w={}]ABC $is $to $air <>subject:Lost."

        assert equal_queries(q1, q2, spike_annotator) is True
        assert equal_queries(q1, q3, spike_annotator) is False
        assert equal_queries(q2, q3, spike_annotator) is False
