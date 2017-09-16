# -*- coding: utf-8 -*-

import requests
import json
from .utils import ascii_alphafy, RateLimitError
import sys


class GoogleETL:
  def __init__(self, db):
    self.db = db
    self.google_search_api_key = open('keys/google_search_api_key').read().strip()
    self.search_engine_id = open('keys/search_engine_id').read().strip()

  def extract(self, title):
    urlified_title = "+".join(ascii_alphafy(title).split(" "))
    url = "https://www.googleapis.com/customsearch/v1?q={}&cx={}&key={}".format(urlified_title, self.search_engine_id, self.google_search_api_key)
    response = requests.get(url)
    if response.status_code >= 400:
      print response.status_code
      print response.text
      raise RateLimitError
    return response

  def transform(self, nips_paper_id, response):
    body = json.loads(response.text)
    results = body.get('items') or []
    truncated_search_query = body['queries']['request'][0]['searchTerms'][:40]  # Long titles are truncated in results

    abstract = {
      "pdf_url": None,
      "abstract_url": None,
      "nips_paper_id": nips_paper_id
    }

    try:
      matches = [result for result in results if truncated_search_query in ascii_alphafy(result['title'])]
    except UnicodeEncodeError:
      return abstract

    if len(matches):
      match = matches[0]
      url = match["link"]
      assert "arxiv.org" in url
      if "pdf" in url:
        abstract["pdf_url"] = url
        abstract["abstract_url"] = url.replace("pdf", "abs")
      else:
        abstract["abstract_url"] = url
        abstract["pdf_url"] = url.replace("abs", "pdf")
    else:
      print "----not found in----\n%s\n" % "\n".join(["\t" + r['title'] for r in results])

    return abstract

  def load(self, record):
    self.db.upsert_search_result(record)
