import requests
import json
from .utils import ascii_alphafy


class GoogleETL:
  def __init__(self, db):
    self.db = db
    self.google_search_api_key = open('keys/google_search_api_key').read().strip()
    self.search_engine_id = open('keys/search_engine_id').read().strip()

  def extract(self, title):
    urlified_title = "+".join(ascii_alphafy(title).split(" "))
    url = "https://www.googleapis.com/customsearch/v1?q={}&cx={}&key={}".format(urlified_title, self.search_engine_id, self.google_search_api_key)
    print "requesting %s" % url
    response = requests.get(url)
    if response.status_code >= 400:
      print response.status_code
      print response.text
      from IPython import embed; embed()
    return response

  def transform(self, response):
    body = json.loads(response.text)
    results = body.get('items')
    first_result = results[0]
    truncated_search_query = body['queries']['request'][0]['searchTerms'][:40]  # Long titles are truncated in results
    if truncated_search_query in ascii_alphafy(first_result["title"]):
      url = first_result["link"]
      assert "arxiv.org" in url
      matching_result = {}
      if "pdf" in url:
        matching_result["pdf_url"] = url
        matching_result["abstract_url"] = url.replace("pdf", "abs")
      else:
        matching_result["abstract_url"] = url
        matching_result["pdf_url"] = url.replace("abs", "pdf")
      return matching_result

  def load(self, nips_paper_id, record):
    record["nips_paper_id"] = str(nips_paper_id)
    self.db.upsert_search_result(record)
