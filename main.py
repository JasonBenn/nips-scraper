# -*- coding: utf-8 -*-

import requests
import sys
from argparse import ArgumentParser
import io
import json
from src.db import DB
from src.arxiv_etl import ArxivETL
from src.google_etl import GoogleETL
from src.nips_etl import NipsETL
from src.utils import RateLimitError


NUM_NIPS_17_PAPERS = 680

def scrape(start_index):
  db = DB()
  nips = NipsETL(db)
  google = GoogleETL(db)
  arxiv = ArxivETL(db)

  titles = db.all('nips_papers')
  print "found %s nips_papers" % len(titles)
  if len(titles) < NUM_NIPS_17_PAPERS:
    print "fetching..."
    response = nips.extract()
    titles = nips.transform(response)
    nips.load(titles)

  all_nips_papers_missing_abstracts = db.all_nips_papers_missing_abstracts()
  print "found %i nips papers missing abstracts" % len(all_nips_papers_missing_abstracts)

  for record in all_nips_papers_missing_abstracts:
    print "fetching #%d: %s" % (record['id'], record['title'])
    try:
      google_response = google.extract(record["title"])
    except RateLimitError:
      break
    search_result = google.transform(record['id'], google_response)
    google.load(search_result)

    if search_result["abstract_url"]:
      print "found search result!"
      arxiv_response = arxiv.extract(search_result["abstract_url"])
      abstract = arxiv.transform(arxiv_response)
      arxiv.load(record["id"], abstract)

  db.to_md("abstracts.md")

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--start-index", nargs='?', default=0)
  args = parser.parse_args()
  start_index = int(args.start_index)
  scrape(start_index)

# INSERT INTO nips_papers (title) VALUES ('sup');

# INSERT INTO google_search_results (nips_paper_id, abstract_url, pdf_url, fetch_attempts)
# VALUES (5, 'abs_url', 'pdf_url', 1)
# ON CONFLICT (nips_paper_id) DO UPDATE SET fetch_attempts =
#   (SELECT fetch_attempts from google_search_results WHERE nips_paper_id = 1) + 1;
