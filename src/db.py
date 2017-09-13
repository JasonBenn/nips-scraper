import sqlite3
import io


class DB:
  def __init__(self):
    self.conn = sqlite3.connect('db.sqlite')
    self.cursor = self.conn.cursor()
    self.load_schema()

  def load_schema(self):
    self.cursor.executescript('''
      CREATE TABLE IF NOT EXISTS nips_papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT UNIQUE ON CONFLICT IGNORE
      );

      CREATE TABLE IF NOT EXISTS google_search_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nips_paper_id INTEGER REFERENCES nips_papers(id),
        abstract_url TEXT,
        pdf_url TEXT,
        fetch_attempts NUMBER
      );

      CREATE TABLE IF NOT EXISTS abstracts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nips_paper_id INTEGER REFERENCES nips_papers(id),
        abstract TEXT
      );
    ''')

  def all(self, table):
    return self.cursor.execute('select * from %s;' % table).fetchall()

  def insert(self, table, records):
    for record in records:
      keys = ", ".join(record.keys())
      values = ", ".join(record.values())
      self.cursor.execute('INSERT INTO %s ("%s") VALUES (?);' % (table, keys), (values,))
    self.conn.commit()

  def to_md(self, filename):
    with io.open(filename, encoding="utf8") as f:
      from IPython import embed; embed()
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
