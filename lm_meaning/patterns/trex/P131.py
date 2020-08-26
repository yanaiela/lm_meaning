from lm_meaning.rules.rule_matching import RuleMatcher


class P131(RuleMatcher):

    def __init__(self):
        search_query = "wikipedia location {}"
        super().__init__(search_query)

    def match_rules(self, line, params={}):

        if 'subdistrict' in line or 'district' in line:
            ans = self.parse_subdistrict(line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'district|subdistrict', 'evidence': line}
        if 'in' in line:
            ans = self.parse_location(line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'is in district of', 'evidence': line}

        return {'answer': ''}

    def parse_location(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['in']:
                if w.head.head.lemma_ == 'be':
                    for c in w.children:
                        if c.dep_ == 'pobj' and c.pos_ == 'PROPN':
                            if c.lemma_.lower() == 'district':
                                for cc in c.children:
                                    if cc.lemma_ == 'of':
                                        for ccc in cc.children:
                                            if ccc.dep_ == 'pobj' and ccc.pos_ == 'PROPN':
                                                return ccc.text
                            return c.text
        return None

    def parse_subdistrict(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['subdistrict', 'district']:
                if w.head.lemma_ in ['be', 'in']:
                    for c in w.children:
                        if c.text == 'of':
                            for cc in c.children:
                                if cc.dep_ == 'pobj':
                                    all_same = True
                                    s_ind = 99999
                                    for ccc in cc.children:
                                        if ccc.dep_ not in ['punct', 'compound']:
                                            all_same = False
                                        else:
                                            if ccc.i < s_ind:
                                                s_ind = ccc.i
                                    if all_same:
                                        return doc[s_ind: cc.i + 1].text
                                    return cc.text
        return None

