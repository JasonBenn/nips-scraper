# NIPS 2017 Scraper

[NIPS 2017 Accepted Papers](https://nips.cc/Conferences/2017/AcceptedPapersInitial) have been released! IT HAS BEGUN.

This script scrapes Arxiv for their abstracts and collects them into [abstracts.md](https://github.com/JasonBenn/nips-scraper/blob/master/abstracts.md).


If you'd like to run it yourself (you're going to need a Google Custom Search API account connected to a billing account), run it like so:
```
python main.py
```

If you'd like to resume progress at a certain point, you can pass the `--start-index` option:
```
python main.py --start-index=3
```
