from lm_meaning.rules.rule_matching import RuleMatcher


class P103(RuleMatcher):

    def __init__(self):
        search_query = "wikipedia {}"
        super().__init__(search_query)

    def match_rules(self, line, params={}):

        if 'language' in line:
            ans = self.parse_language(line)
            if ans is not None:
                return ans
        if 'speak' in line:
            ans = self.parse_speak(line)
            if ans is not None:
                return ans
        if 'is a' in line:
            ans = self.parse_is_a(line)
            if ans is not None:
                return ans

        return None

    def parse_language(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['language']:
                for c in w.children:
                    if c.dep_ == 'appos' and c.pos_ == 'PROPN':
                        return c.text
        return None

    def parse_speak(self, text):
        """used in: https://en.wikipedia.org/wiki/Ngalop_people"""
        doc = self.nlp(text)

        for w in doc:
            if w.lemma_ == 'speak':
                for c in w.children:
                    if c.dep_ == 'dobj' and c.pos_ == 'PROPN':
                        return c.text
        return None

    def parse_is_a(self, text):
        doc = self.nlp(text)

        for w in doc:
            if w.text == 'is':
                for c in w.children:
                    if c.dep_ == 'attr':
                        for cc in c.children:
                            if cc.dep_ == 'amod' and cc.pos_ == 'ADJ':
                                return cc.text
        return None
