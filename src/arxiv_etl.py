class ArxivETL:
  def __init__(self, db):
    self.db = db

  def extract(self, abstract_url):
    return requests.get(abstract_url)

  def transform(self, response):
    arxiv_abstract_html = BeautifulSoup(response.text, "html.parser")
    abstract_text = arxiv_abstract_html.select_one('.abstract').text
    return abstract_text.replace('\n', ' ').replace('Abstract:', '').strip().encode("utf8")

  def load(self, nips_paper_id, abstract):
    record["nips_paper_id"] = nips_paper_id
    self.db.insert("abstracts", { "abstract": abstract })
