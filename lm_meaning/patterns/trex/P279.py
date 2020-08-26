from lm_meaning.rules.rule_matching import RuleMatcher


class P279(RuleMatcher):

    def __init__(self):
        search_query = "wikipedia {}"
        super().__init__(search_query)

    def match_rules(self, line, params={}):

        title = params['title']
        if title in line:
            ans = self.parse_subclass_of(title, line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'title', 'evidence': line}
        if 'is a' in line:
            ans = self.parse_is_a(line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'is a', 'evidence': line}

        return {'answer': ''}

    def parse_subclass_of(self, title, text):
        doc = self.nlp(text)
        for w in doc:
            if w.text == title:
                if w.dep_ == 'compound':
                    if w.head.pos_ == 'NOUN':
                        return w.head.text
        return None

    def parse_is_a(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.text == 'is':
                for c in w.children:
                    if c.dep_ == 'attr':
                        if c.pos_ == 'NOUN':
                            return c.text
        return None

