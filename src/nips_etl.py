import requests
import json
from bs4 import BeautifulSoup


class NipsETL:
  def __init__(self, db):
    self.db = db

  def extract(self):
    response = requests.get("https://nips.cc/Conferences/2017/AcceptedPapersInitial")
    if response.status_code == 200:
      return response

  def transform(self, response):
    parsed_html = BeautifulSoup(response.text, "html.parser")
    return [title.text for title in parsed_html.select("p > b")]

  def load(self, titles):
    self.db.insert('nips_papers', [{ "title": title } for title in titles])
