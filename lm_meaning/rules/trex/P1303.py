from lm_meaning.rules.rule_matching import RuleMatcher


# someone playing and instrument Y
class P1303(RuleMatcher):

    def __init__(self):
        search_query = "wikipedia musician {}"
        super().__init__(search_query)

    def match_rules(self, line, params={}):
        if 'infobox' in params:
            ans = self.parse_infobox(params['infobox'])
            if ans is not None:
                return {'answer': ans, 'explanation': 'infobox', 'rule': 'infobox Instrument|Instrument|Occupation(s)',
                        'evidence': params['infobox']}
        if 'play' in line:
            ans = self.parse_play(line)
            if ans is not None:
                return {'answer': ans, 'explanation': 'rule', 'rule': 'play', 'evidence': line}

        return {'answer': ''}

    def parse_play(self, text):
        doc = self.nlp(text)
        for w in doc:
            if w.lemma_ in ['play']:
                for c in w.children:
                    if c.dep_ == 'dobj':  # and c.pos_ == 'NOUN':
                        for cc in c.children:
                            if cc.dep_ == 'compound' and cc.pos_ == 'NOUN':
                                return cc.text
                        return c.text
        return None

    def parse_infobox(self, infobox):

        for entry in infobox:
            if entry['key'] in ['Instrument', 'Instruments']:
                first_entity = self.nlp(entry['values'][0].lower())
                if len(first_entity) > 1:
                    ans = first_entity[0].head.text
                else:
                    ans = first_entity[0].text
                if ans in ['Vocals', 'vocals', 'vocal']:
                    return 'voice'
                else:
                    return ans
            elif entry['key'] == 'Occupation(s)':
                # singer - https://en.wikipedia.org/wiki/Annalisa
                first = entry['values'][0]
                if first in ['Singer', 'Rapper']:
                    return 'voice'
        return None
