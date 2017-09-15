import psycopg2
import io


class DB:
  def __init__(self):
    self.conn = psycopg2.connect("dbname=nips_scraper user=jasonbenn")
    self.cursor = self.conn.cursor()
    self.load_schema()

  def load_schema(self):
    self.cursor.execute('''
      CREATE TABLE IF NOT EXISTS nips_papers (
        id SERIAL PRIMARY KEY,
        title TEXT UNIQUE
      );

      CREATE TABLE IF NOT EXISTS google_search_results (
        id SERIAL PRIMARY KEY,
        nips_paper_id INTEGER UNIQUE REFERENCES nips_papers,
        abstract_url TEXT,
        pdf_url TEXT,
        fetch_attempts INTEGER
      );

      CREATE TABLE IF NOT EXISTS abstracts (
        id SERIAL PRIMARY KEY,
        nips_paper_id INTEGER UNIQUE REFERENCES nips_papers,
        abstract TEXT,
        authors TEXT,
        category TEXT
      );
    ''')

  def all(self, table):
    return self.cursor.execute('''SELECT * FROM ${table};''', { "table": table }).fetchall()

  def all_google_search_results(self, table):
    return self.cursor.execute('''
      SELECT * FROM nips_papers
      LEFT JOIN abstracts ON abstracts.nips_paper_id=nips_papers.id
      JOIN google_search_results ON google_search_results.nips_paper_id=nips_papers.id
      WHERE abstract IS NULL
      ORDER BY fetch_attempts ASC;
    ''').fetchall()

  def insert(self, table, records):
    for record in records:
      keys = ", ".join(record.keys())
      values = ", ".join(record.values())
      self.cursor.execute('INSERT INTO %s ("%s") VALUES (?);' % (table, keys), (values,))
    self.conn.commit()

  def upsert_search_result(self, table, record):
    self.cursor.execute('''
      INSERT INTO google_search_results (nips_paper_id, abstract_url, pdf_url, fetch_attempts)
      VALUES (${nips_paper_id}, ${abstract_url}, ${pdf_url}, 1)
      ON CONFLICT UPDATE (fetch_attempts)
      VALUES (
        (SELECT fetch_attempts from google_search_results WHERE nips_paper_id = ${nips_paper_id}) + 1
      )
    ''', record)

  def to_md(self, filename):
    print "to_md"
    # with io.open(filename, encoding="utf8") as f:
      # from IPython import embed; embed()
      # f.write(u"### #%s: %s\n" % (i + start_index, title))
      # # extract, transform
      # abstract = arxiv_url_to_abstract(arxiv_url)
      # # load
      # f.write(u"[%s](%s)\n" % (arxiv_url, arxiv_url))
      # f.write(abstract + u"\n\n")
      # f.flush()

      # try:
      #   assert title[:40] in result["title"]  # Long titles are truncated in results
      # except AssertionError:
      #   f.write(u"Not found on arxiv\n\n")
      #   f.flush()
