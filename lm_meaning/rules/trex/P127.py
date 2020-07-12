from lm_meaning.rules.rule_matching import RuleMatcher


# "completed the acquisition":
# https://en.wikipedia.org/wiki/3Com
class P127(RuleMatcher):

    def __init__(self):
        search_query = "{} wikipedia"
        super().__init__(search_query)

    def match_rules(self, line, params={}):
        if 'infobox' in params:
            ans = self.parse_infobox(params['infobox'])
            if ans is not None:
                return {'answer': ans, 'explanation': 'infobox', 'rule': 'Parent|Owner|Successor|Owners',
                        'evidence': params['infobox']}
        if 'acquire' in line:
            ans = self.parse_aquired(line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'acquire by', 'evidence': line}

        if 'part of' in line:
            ans = self.parse_became_part_of(line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'part of', 'evidence': line}

        return {'answer': ''}

    def parse_aquired(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['acquire']:
                for c in w.children:
                    if c.dep_ == 'agent' and c.text == 'by':
                        for cc in c.children:
                            if cc.dep_ == 'pobj' and cc.pos_ == 'PROPN':
                                return cc.text
        return None

    def parse_became_part_of(self, text):
        # succeeded only 1 example... consider removing
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['become']:
                for c in w.children:
                    if c.text == 'part':
                        for cc in c.children:
                            if cc.text == 'of':
                                for ccc in cc.children:
                                    if ccc.pos_ == 'PROPN':
                                        return ccc.text

    def parse_infobox(self, infobox):
        # parent in the infobox: https://en.wikipedia.org/wiki/Archambault
        # Owner in the infobox: https://en.wikipedia.org/wiki/Vizada,
        # also: https://en.wikipedia.org/wiki/SerbianTV-America
        # Successor in the infobox: https://en.wikipedia.org/wiki/3Com
        # Owners: https://en.wikipedia.org/wiki/Denso
        for entry in infobox:
            if entry['key'] in ['Parent', 'Owner', 'Successor', 'Owners']:
                return entry['values'][-1]
        return None
