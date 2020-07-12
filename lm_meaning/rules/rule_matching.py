import spacy
from tqdm import tqdm
from lm_meaning.rules.utils import search_results, get_text_from_url
import os
import jsonlines
from pathlib import Path


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
            # in case the search result did not find a relevant url
            if url is None:
                entry_result['answer'] = 'no-url'
                entry_result['evidence'] = None
                instances_output.append(entry_result)
                print('no url')
                continue

            title = url.split('/')[-1]
            params = {'title': title}

            html_data = get_text_from_url(url)
            url_text = html_data['text']
            params['infobox'] = html_data['infobox']
            ans = None
            for line in url_text.split('\n'):
                if line.strip() == '':
                    continue
                ans = self.match_rules(line, params)
                if ans['answer']:
                    entry_result.update(ans)
                    # entry_result['ans'] = ans
                    # entry_result['evidence'] = line
                    break

            if not ans['answer']:
                entry_result['answer'] = ''
                entry_result['evidence'] = ''

            instances_output.append(entry_result)

        return instances_output

    def match_rules(self, line, params={}):
        raise NotImplemented

    def persist_answers(self, instances, in_dir):
        f_name = self.__class__.__name__
        Path(in_dir).mkdir(parents=True, exist_ok=True)

        with jsonlines.open(os.path.join(in_dir, f_name) + '.jsonl', 'w') as f:
            f.write_all(instances)
