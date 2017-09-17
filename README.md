# NIPScraper - scraper for NIPS 2017 paper abstracts

Like most people I'm just trying to figure out how to best spend my time at this conference. Figured that collecting and reading [all of the abstracts](https://nips.cc/Conferences/2017/AcceptedPapersInitial) (680 of them!) would help me find the most relevant talks/posters/workshops for my work. I'm hoping that reading all this will make it easier to identify some promising research directions, too.

Most of the papers aren't on the first page of Google results, which I take to mean that they haven't been published yet, and I'm also working around a daily rate limit of a few hundred searches from Google's API. Currently I have 69/680.

Watch or star this repo, I'll be updating it frequently/daily! Contributions/feature requests welcome, of course.

This program scrapes Arxiv for paper abstracts, authors, categories, dumps them into a [Postgres database](https://github.com/JasonBenn/nips-scraper/blob/master/dump.sql), and exports them to [abstracts.md](https://github.com/JasonBenn/nips-scraper/blob/master/abstracts.md) (which is nicer for reading on a Kindle).


## Related...

I also made some simple JS one-liners that copy NIPS workshop and tutorial information to your clipboard so that you can paste them into a spreadsheet. Find the spreadsheet [here](https://docs.google.com/spreadsheets/d/1gQpSSjoypqtTSPaJdLvT8UsGEgjJXZSZc0KkLlSDLFk/edit?usp=sharing) (the snippets are saved as comments).


## Contributing

Any pull requests are welcome, of course.

If you'd like to run it yourself, you'll need a Google Custom Search API account connected to a billing account and postgres 9.6+. Run like so:
```
python main.py
```
