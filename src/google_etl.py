import requests
import json
from .utils import clean_text


class GoogleETL:
  def __init__(self, db):
    self.db = db
    self.google_search_api_key = open('keys/google_search_api_key').read().strip()
    self.search_engine_id = open('keys/search_engine_id').read().strip()

  def extract(self, title):
    urlified_title = "+".join(clean_text(title).split(" "))
    url = "https://www.googleapis.com/customsearch/v1?q={}&cx={}&key={}".format(urlified_title, self.search_engine_id, self.google_search_api_key)
    print "requesting %s" % url
    response = requests.get(url)
    if response.status_code >= 400:
      print response.text
      from IPython import embed; embed()
    return response

  def transform(self, response):
    body = json.loads(response.text)
    results = body.get('items')
    first_result = results[0]
    search_query = body['queries']['request'][0]['searchTerms']
    truncated_search_query = search_query[:40]  # Long titles are truncated in results
    if truncated_search_query in clean_text(first_result["title"]):
      url = first_result["link"]
      result = {}
      if "pdf" in url:
        result["pdf_url"] = url
        result["abstract_url"] = url.replace("pdf", "abs")
      else:
        result["abstract_url"] = url
        result["pdf_url"] = url.replace("abs", "pdf")
      return result

  def load(self, nips_paper_id, record):
    record["nips_paper_id"] = str(nips_paper_id)
    self.db.upsert_search_result(record)
