from lm_meaning.spike.utils import lexical_difference

import spacy


class TestPluralInstruction:
    nlp = spacy.load('en_core_web_sm')

    def test_equal_queries(self):

        q1 = "[Y] debuted [X]."
        q2 = "[X] debuted on [Y]."
        q3 = "[X] aired on [Y]."
        q4 = "[X] was originally aired on [Y]."

        result = lexical_difference(q1, q2, self.nlp)
        assert result['diff_lemma'] is False
        assert result['diff_det'] is False

        result = lexical_difference(q1, q3, self.nlp)
        assert result['diff_lemma'] is True
        assert result['diff_det'] is False

        result = lexical_difference(q3, q4, self.nlp)
        assert result['diff_lemma'] is True
        assert result['diff_det'] is False

    def test_determiners(self):
        q1 = "In [X] [Y] is an official language."
        q2 = "The official language of [X] is [Y]."
        q3 = "The official language of [X] is the [Y] language."

        result = lexical_difference(q1, q2, self.nlp)
        assert result['diff_lemma'] is False
        assert result['diff_det'] is True

        # words repeating themselves
        result = lexical_difference(q2, q3, self.nlp)
        assert result['diff_lemma'] is True
        assert result['diff_det'] is True

    def test_lemmas(self):
        q1 = "[X] died in [Y]."
        q2 = "[X] died at [Y]."

        result = lexical_difference(q1, q2, self.nlp)
        assert result['diff_lemma'] is True
        assert result['diff_det'] is False
