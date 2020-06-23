import json
import os
from functools import lru_cache
from pathlib import Path

import requests
from azure.cognitiveservices.search.websearch import WebSearchClient
from bs4 import BeautifulSoup
from msrest.authentication import CognitiveServicesCredentials


def read_key():
    with open('subscription.json', 'r') as f:
        line = f.read().strip()
    json_data = json.loads(line)
    return json_data['azure_key']


subscription_key = read_key()
# Instantiate the client and replace with your endpoint.
azure_client = WebSearchClient(endpoint="https://eastus.api.cognitive.microsoft.com/",
                               credentials=CognitiveServicesCredentials(subscription_key))


def read_file(in_file):
    with open(in_file, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(x.strip()) for x in lines]
    entities = [{'obj': x['obj_label'], 'sub': x['sub_label']} for x in lines]
    return entities


def read_cache(in_file):
    if not os.path.isfile(in_file):
        return {}
    with open(in_file, 'r') as f:
        data = f.read()
    json_data = json.loads(data)
    return json_data


def save_cache(out_file, json_obj):
    Path(out_file.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(json_obj, f)


def cache_web_queries(cache_path=str(Path.home())):
    cache_file = cache_path + '/.web_queries/query.json'

    def decorate(func):
        def call(*args, **kwargs):
            json_data = read_cache(cache_file)
            query = args[0]
            if query in json_data:
                result = json_data[query]
            else:
                result = func(*args, **kwargs)
                json_data[query] = result
                save_cache(cache_file, json_data)
            return result
        return call
    return decorate


@cache_web_queries()
@lru_cache(maxsize=None)
def search_results(query):

    # Make a request.
    web_data = azure_client.web.search(query=query)

    '''
    Web pages
    If the search response contains web pages, the first result's name and url
    are printed.
    '''
    if hasattr(web_data.web_pages, 'value'):

        web_page = web_data.web_pages.value[0]

        # skipping the simple wikipedia result
        if 'simple.wikipedia' in web_page.url and len(web_data.web_pages.value) > 1:
            web_page = web_data.web_pages.value[1]

        return web_page.url

    return None


def cache_url_fetch(cache_path=str(Path.home())):
    cache_file = cache_path + '/.web_queries/urls.json'

    def decorate(func):
        def call(*args, **kwargs):
            json_data = read_cache(cache_file)
            url = args[0]
            if url in json_data:
                text = json_data[url]
            else:
                text = func(*args, **kwargs)
                json_data[url] = text
                save_cache(cache_file, json_data)
            return text
        return call
    return decorate


def read_infobox(text):
    soup = BeautifulSoup(text, 'lxml')
    table_content = []
    infobox = soup.find('table', class_='infobox')
    if not infobox:
        return table_content
    for items in infobox.find_all('tr'):
        data = items.find_all(['th', 'td'])
        try:
            title = data[0].text
            answer = data[1]
        except IndexError:
            continue
        row_dic = {'key': title}

        answer_values = []
        if answer.text:
            for split in answer.stripped_strings:
                for comma_sep in split.split(','):
                    answer_values.append(comma_sep.strip())

        for x in answer.find_all('a'):
            answer_values.append(x.text)

        row_dic['values'] = answer_values
        table_content.append(row_dic)
    return table_content


# @lru_cache(maxsize=None)
@cache_url_fetch()
def get_text_from_url(url):
    resp = requests.get(url)
    txt = resp.text
    soup = BeautifulSoup(txt, 'html.parser')

    table_content = read_infobox(txt)
    # filtering some elements for a clearer text.

    # filtering tables
    for table in soup.find_all("table"):
        table.extract()

    # filtering subscripts, which mess up the parsing sometimes (e.g. https://en.wikipedia.org/wiki/Kissyfur)
    for sup in soup.find_all("sup"):
        sup.extract()

    return {'text': soup.get_text(), 'infobox': table_content}


def eval_performance(entry_answers):
    acc = 0
    for entry in entry_answers:
        if entry['ans'] == entry['obj']:
            acc += 1
    return acc / len(entry_answers)
