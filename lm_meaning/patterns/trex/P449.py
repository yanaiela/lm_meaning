from lm_meaning.patterns.rule_matching import RuleMatcher


class P449(RuleMatcher):

    def __init__(self):
        search_query = "wikipedia serie {}"
        super().__init__(search_query)

    def match_rules(self, text, obj, params={}):
        rule_answers = []
        found_rules = set()

        for line in text:
            if not line: continue

            if 'premiere on' not in found_rules and 'premier' in line:
                ans = self.parse_series_text(line, 'premiere', preposition='on')
                if ans == obj:
                    rule_answers.append({'answer': ans, 'explanation': 'rule', 'rule': 'premiere on', 'evidence': line})
                    found_rules.add('premiere on')
            if 'air on' not in found_rules and 'air' in line:
                ans = self.parse_series_text(line, 'air', preposition='on')
                if ans == obj:
                    rule_answers.append({'answer': ans, 'explanation': 'rule', 'rule': 'air on', 'evidence': line})
                    found_rules.add('air on')
            if 'broadcast on' not in found_rules and 'broadcast' in line:
                ans = self.parse_series_text(line, 'broadcast', preposition='on')
                if ans == obj:
                    rule_answers.append({'answer': ans, 'explanation': 'rule', 'rule': 'broadcast on', 'evidence': line})
                    found_rules.add('broadcast on')

        return rule_answers

    def parse_series_text(self, text, verb_lemma, preposition):
        """
        The 'broadcast_on' rule was not part of the automated generated rules, but noticed from the data
        specifically - https://en.wikipedia.org/wiki/Homeland_(TV_series)
        """
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ == verb_lemma:
                for c in w.children:
                    if c.text == preposition:
                        for cc in c.children:
                            if cc.ent_type_ in ['DATE']:
                                continue

                            # dealing with nouns
                            # based on the following serie:
                            # https://en.wikipedia.org/wiki/Cuts_(TV_series)
                            # (Cuts is an American sitcom that aired on the UPN network from February 14, 2005,
                            # to May 11, 2006, and is a spin-off of another UPN series, One on One.)
                            # ignoring the noun, and looking for the proper noun
                            if cc.pos_ in ['NOUN']:
                                for ccc in cc.children:
                                    if ccc.dep_ in ['compound'] and ccc.pos_ in ['PROPN']:
                                        return ccc.text
                            return cc.text
        return None
