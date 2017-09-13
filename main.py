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


def scrape(start_index):
  db = DB()
  nips = NipsETL(db)
  google = GoogleETL(db)
  arxiv = ArxivETL(db)

  titles = db.all("nips_papers")
  if not len(titles):
    response = nips.extract()
    titles = nips.transform(response)
    nips.load(titles)

  for i, title in titles[start_index:][3:4]:
    response = google.extract(title)
    search_result = google.transform(response)
    print "search_result:"
    print search_result
    if search_result:
      print "found search_result"
      google.load(i, search_result)
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
