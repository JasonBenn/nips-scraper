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
        title TEXT UNIQUE ON CONFLICT DO NOTHING
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
    return self.cursor.execute("SELECT * FROM %s;", (table,)).fetchall()

  def all_nips_papers_missing_abstracts(self, table):
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
      self.cursor.execute("INSERT INTO %(table) ($(keys)) VALUES (%(values));", { "table": table, "keys": keys, "values": values })
    self.conn.commit()

  def upsert_search_result(self, table, search_result):
    self.cursor.execute('''
      INSERT INTO google_search_results (nips_paper_id, abstract_url, pdf_url, fetch_attempts)
      VALUES (%(nips_paper_id), %(abstract_url), %(pdf_url), 1)
      ON CONFLICT UPDATE (fetch_attempts)
      VALUES (
        (SELECT fetch_attempts from google_search_results WHERE nips_paper_id = %(nips_paper_id)) + 1
      )
    ''', search_result)

  def to_md(self, filename):
    print "dumping db to md"

    abstracts = self.cursor.execute('''
      SELECT * FROM nips_papers
      LEFT JOIN abstracts ON abstracts.nips_paper_id=nips_papers.id
    ''').fetchall()

    print "found %i abstracts" % len(abstracts)

    with io.open(filename, encoding="utf8") as f:
      for a in abstracts:
        f.write(u"### #%s: %s\n" % (a["nips_papers.id"], a["title"]))
        f.write(u"[Abstract](%s), [PDF](%s)\n" % (a["abstract_url"], a["pdf_url"]))
        f.write(u"Authors: %s" % a["authors"])
        f.write(a["abstract"] + u"\n\n")
