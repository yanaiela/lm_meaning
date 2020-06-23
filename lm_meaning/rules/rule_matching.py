import spacy
from tqdm import tqdm
from lm_meaning.rules.utils import search_results, get_text_from_url


class RuleMatcher:

    def __init__(self, search_query):
        self.nlp = spacy.load("en_core_web_sm")
        self.search_query = search_query

    def process_relation(self, data, subset=10):

        assert self.search_query is not None

        instances_output = []

        filtered_data = data
        if subset:
            filtered_data = data[:subset]
        for entry in tqdm(filtered_data):
            subj = entry['sub']
            obj = entry['obj']

            entry_result = {'sub': subj, 'obj': obj}

            url = search_results(self.search_query.format(subj))
            entry_result['url'] = url
            if url is None:
                print('no url')
                continue

            title = url.split('/')[-1]
            params = {'title': title}

            url_text = get_text_from_url(url)
            ans = None
            for line in url_text.split('\n'):
                if line.strip() == '':
                    continue
                ans = self.match_rules(line, params)
                if ans is not None:
                    entry_result['ans'] = ans
                    entry_result['evidence'] = line
                    break

            if not ans:
                entry_result['ans'] = None
                entry_result['evidence'] = None

            instances_output.append(entry_result)

        return instances_output

    def match_rules(self, line, params={}):
        raise NotImplemented
