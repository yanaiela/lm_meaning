import re

from lm_meaning.rules.rule_matching import RuleMatcher


class P37(RuleMatcher):

    def __init__(self):
        search_query = "wikipedia location {}"
        super().__init__(search_query)

    def match_rules(self, line, params={}):
        line = re.sub(r' \([^)]*\)', '', line)
        line = re.sub(r'\([^)]*\) ', '', line)
        line = re.sub(r'\([^)]*\)', '', line)

        if 'speak' in line:
            ans = self.parse_speaks(line)
            if ans is not None:
                return ans

        return None

    def parse_speaks(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['speak']:
                for c in w.children:
                    if c.dep_ == 'dobj':
                        return c.text
        return None
