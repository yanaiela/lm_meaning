from lm_meaning.spike.utils import equal_queries, get_spike_objects
from spike.spacywrapper.annotator import SpacyAnnotator


class TestPluralInstruction:
    def test_equal_queries(self):
        # _, spike_annotator = get_spike_objects()
        spike_annotator = SpacyAnnotator.from_config("en.json")

        q1 = "<>subject:Lost $was $aired $on object:[w={}]ABC"
        q2 = "<>subject:Lost $was $premiered $on object:[w={}]ABC"
        q3 = "object:[w={}]ABC $is $to $air <>subject:Lost."

        assert equal_queries(q1, q2, spike_annotator) is True
        assert equal_queries(q1, q3, spike_annotator) is False
        assert equal_queries(q2, q3, spike_annotator) is False

    def test_specific(self):
        _, spike_annotator = get_spike_objects()

        q1 = "object:[w={}]ABC $is $to $broadcast <>subject:Lost"
        q2 = "object:[w={}]ABC $is $to $air <>subject:Lost"

        assert equal_queries(q1, q2, spike_annotator) is True

    def test_easy(self):
        spike_annotator = SpacyAnnotator.from_config("en.json")

        q1 = "<>subject:John $died $in object:[w={}]England."
        q2 = "<>subject:John $died $at object:[w={}]England."

        assert equal_queries(q1, q2, spike_annotator) is True
