import os
import urllib.request
import pandas as pd


class WebScraper:

    def __init__(self, url_base="https://www.treasurydirect.gov/", url_extension="auctions/auction-query/"):
        """ Initializes the web-scraper object with the provided url."""
        self.url = os.path.join(url_base, url_extension)
        self.html_str = None
        r"https://www.treasurydirect.gov/WS/common/export/Securities.csv"

    def get_url(self):
        return self.url

    def read_url_into_html_str(self) -> str:
        web_page = urllib.request.urlopen(self.url)
        page_in_html_bytes = web_page.read()
        self.html_str = page_in_html_bytes.decode("utf-8")



