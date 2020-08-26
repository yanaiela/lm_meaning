import spacy
from tqdm import tqdm
from lm_meaning.rules.utils import search_results, get_text_from_url
import os
import jsonlines
from pathlib import Path
from copy import deepcopy


class RuleMatcher:

    def __init__(self, search_query):
        self.nlp = spacy.load("en_core_web_sm")
        self.search_query = search_query

    def process_relation(self, data, subset=10):

        assert self.search_query is not None

        instances_output = []

        filtered_data = data
        if subset:
            filtered_data = data[100:100+subset]
        for entry in tqdm(filtered_data):
            subj = entry['sub']
            obj = entry['obj']

            entry_results = {'sub': subj, 'obj': obj, 'answers': []}

            url = search_results(self.search_query.format(subj))
            entry_results['url'] = url

            base_result = {'sub': subj, 'obj': obj}
            # in case the search result did not find a relevant url
            if url is None:
                base_result['answer'] = ''
                base_result['evidence'] = 'no-url'
                entry_results['answers'].append(base_result)

                instances_output.append(entry_results)
                print('no url')
                continue

            title = url.split('/')[-1]
            params = {'title': title}

            html_data = get_text_from_url(url)
            url_text = html_data['text']
            params['infobox'] = html_data['infobox']

            answers = self.match_rules(url_text.split('\n'), obj, params)

            for ans in answers:
                entity_copy = deepcopy(base_result)
                entity_copy.update(ans)
                entry_results['answers'].append(entity_copy)

            instances_output.append(entry_results)

        return instances_output

    def match_rules(self, line, obj, params={}):
        raise NotImplemented

    def persist_answers(self, instances, in_dir):
        f_name = self.__class__.__name__
        Path(in_dir).mkdir(parents=True, exist_ok=True)

        with jsonlines.open(os.path.join(in_dir, f_name) + '.jsonl', 'w') as f:
            f.write_all(instances)
