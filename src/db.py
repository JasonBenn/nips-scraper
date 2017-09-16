# -*- coding: utf-8 -*-

import psycopg2
from psycopg2 import sql
import io
from psycopg2.extras import RealDictCursor


class DB:
  def __init__(self):
    self.conn = psycopg2.connect("dbname=nips_scraper user=jasonbenn")
    self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
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
    select_all = sql.SQL("SELECT * FROM {};").format(sql.Identifier(table))
    self.cursor.execute(select_all)
    return self.cursor.fetchall()

  def all_nips_papers_missing_abstracts(self):
    self.cursor.execute('''
      SELECT abstract, abstract_url, authors, category, nips_papers.id, pdf_url, title FROM nips_papers
      LEFT JOIN abstracts ON abstracts.nips_paper_id=nips_papers.id
      LEFT JOIN google_search_results ON google_search_results.nips_paper_id=nips_papers.id
      WHERE abstract IS NULL
      ORDER BY fetch_attempts ASC;
    ''')
    return self.cursor.fetchall()

  def insert_nips_paper(self, title):
    self.cursor.execute("INSERT INTO nips_papers (title) VALUES (%s) ON CONFLICT DO NOTHING;", (title,))
    self.conn.commit()

  def insert_abstract(self, record):
    self.cursor.execute('''
      INSERT INTO abstracts (nips_paper_id, abstract, authors, category)
      VALUES (%(nips_paper_id)s, %(abstract)s, %(authors)s, %(category)s) ON CONFLICT DO NOTHING;
    ''', record)
    self.conn.commit()

  def upsert_search_result(self, search_result):
    try:
      self.cursor.execute('''
        INSERT INTO google_search_results (nips_paper_id, abstract_url, pdf_url, fetch_attempts)
        VALUES (%(nips_paper_id)s, %(abstract_url)s, %(pdf_url)s, 1)
        ON CONFLICT (nips_paper_id) DO UPDATE SET fetch_attempts = (
          (SELECT fetch_attempts from google_search_results WHERE nips_paper_id = %(nips_paper_id)s) + 1
        )
      ''', search_result)
    except KeyError:
      print search_result
    self.conn.commit()

  def to_md(self, filename):
    print "dumping db to md"

    self.cursor.execute('''
      SELECT abstract, abstract_url, authors, category, nips_papers.id, pdf_url, title FROM nips_papers
      LEFT JOIN abstracts ON abstracts.nips_paper_id=nips_papers.id
      LEFT JOIN google_search_results ON google_search_results.nips_paper_id=nips_papers.id
      ORDER BY nips_papers.id ASC;
    ''')
    abstracts = self.cursor.fetchall()

    print "found %i abstracts" % len(abstracts)

    with io.open(filename, 'w', encoding="utf8") as f:
      for a in abstracts:
        f.write(u"### #%s: %s\n" % (a["id"], a["title"].decode('utf8', 'ignore')))
        if a["authors"]:
          f.write(u"_%s_\n\n" % a["authors"].decode('utf8', 'ignore'))
        if a["abstract"]:
          f.write(u"%s\n" % a["abstract"].decode('utf8', 'ignore'))
        if a["abstract_url"]:
          f.write(u"[Abstract](%s), [PDF](%s)\n\n" % (a["abstract_url"], a["pdf_url"]))
        f.write(u"\n")
