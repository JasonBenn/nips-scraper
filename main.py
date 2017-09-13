# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import sys
from argparse import ArgumentParser
import io
import json

GOOGLE_SEARCH_API_KEY = open('keys/google_search_api_key').read().strip()
SEARCH_ENGINE_ID = open('keys/search_engine_id').read().strip()


def title_to_google_results(title):
  urlified_title = "+".join(title.replace(r':', '').replace(',', '').encode('ascii', errors='ignore').split(" "))
  response = requests.get("https://www.googleapis.com/customsearch/v1?q={}&cx={}&key={}".format(urlified_title, SEARCH_ENGINE_ID, GOOGLE_SEARCH_API_KEY))
  return json.loads(response.text)

def parse_abstract_text(abstract_text):
  return abstract_text.replace('\n', ' ').replace('Abstract:', '').strip().encode("utf8")

def arxiv_url_to_abstract(arxiv_url):
  if "pdf" in arxiv_url:
    arxiv_url = arxiv_url.replace("pdf", "abs")
  arxiv_abstract_doc = requests.get(arxiv_url)
  arxiv_abstract_html = BeautifulSoup(arxiv_abstract_doc.text, "html.parser")
  abstract_text = arxiv_abstract_html.select_one('.abstract').text
  return parse_abstract_text(abstract_text)

def nips_papers_titles():
  nips_doc = requests.get("https://nips.cc/Conferences/2017/AcceptedPapersInitial")
  nips_html = BeautifulSoup(nips_doc.text, "html.parser")
  return [title.text for title in nips_html.select("p > b")]

def scrape(start_index):
  titles = nips_papers_titles()
  with io.open("abstracts.md", "a", encoding="utf8") as f:
    for i, title in enumerate(titles[start_index:]):
      f.write(u"### #%s: %s\n" % (i + start_index, title))

      response = title_to_google_results(title)

      try:
        result = response.get('items')[0]
      except TypeError:
        # results.get('items') is prob None -- rate limit?
        f.write(u"%s\n\n" % response)
        f.flush()
        break

      arxiv_url = result["formattedUrl"]

      try:
        assert title[:40] in result["title"]  # Long titles are truncated in results
      except AssertionError:
        f.write(u"Not found on arxiv\n\n")
        f.flush()
        continue

      abstract = arxiv_url_to_abstract(arxiv_url)
      f.write(u"[%s](%s)\n" % (arxiv_url, arxiv_url))
      f.write(abstract + u"\n\n")
      f.flush()

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--start-index", nargs='?', default=0)
  args = parser.parse_args()
  start_index = int(args.start_index)
  scrape(start_index)
