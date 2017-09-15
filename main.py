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

  for i, title in all_nips_papers_missing_abstracts[start_index:]:
    print "fetching #%i: %s" % (i, title)
    response = google.extract(title)
    search_result = google.transform(response)
    google.load(i, search_result)

    if search_result:
      print "found search result!"
      response = arxiv.extract(search_result["abstract_url"])
      abstract = arxiv.transform(response)
      arxiv.load(i, abstract)

  db.to_md("abstracts.md")

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--start-index", nargs='?', default=0)
  args = parser.parse_args()
  start_index = int(args.start_index)
  scrape(start_index)
