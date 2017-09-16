# NIPScraper - scraper for NIPS 2017 Accepted Papers abstracts

[NIPS 2017 Accepted Papers](https://nips.cc/Conferences/2017/AcceptedPapersInitial) were released this week -- hopefully this data will help you figure out which posters to find, which talks to see, and which workshops to attend. Not to worry, there's only... 680 of them.

This program scrapes Arxiv for paper abstracts, authors, categories, dumps them into a [Postgres database](), and exports them to [abstracts.md](https://github.com/JasonBenn/nips-scraper/blob/master/abstracts.md) (which is nice for reading on your Kindle).

Many of these papers haven't been published to Arxiv yet. Watch this repo for updates, I'll be re-scraping and updating frequently.


## Related...

I've also got some simple JS one-liners that copy NIPS workshop and tutorial information to your clipboard so that you can paste them into a spreadsheet. Find it [here]().


## Contributing

Any pull requests are welcome, of course.

If you'd like to run it yourself, you'll need a Google Custom Search API account connected to a billing account and postgres 9.6+. Run like so:
```
python main.py
```

