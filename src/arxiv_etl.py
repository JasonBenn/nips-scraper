# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from utils import strip_text


class ArxivETL:
  def __init__(self, db):
    self.db = db

  def extract(self, abstract_url):
    return requests.get(abstract_url)

  def transform(self, response):
    arxiv_abstract_html = BeautifulSoup(response.text, "html.parser")
    abstract_text = arxiv_abstract_html.select_one('.abstract').text.replace('Abstract:', '')
    authors_text = arxiv_abstract_html.select_one('.authors').text.replace('Authors:', '')
    arxiv_category = arxiv_abstract_html.select_one('.subheader').text
    return {
      "authors": strip_text(authors_text),
      "abstract": strip_text(abstract_text),
      "category": strip_text(arxiv_category)
    }

  def load(self, nips_paper_id, abstract_record):
    abstract_record["nips_paper_id"] = nips_paper_id
    self.db.insert_abstract(abstract_record)
