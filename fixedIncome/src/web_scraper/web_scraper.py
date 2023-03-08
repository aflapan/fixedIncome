import os
import urllib
import pandas
import pandas as pd


class WebScraper:

    def __init__(self, url_base="https://www.treasurydirect.gov/", url_extension="auctions/auction-query/"):
        """ Initializes the web-scraper object with the provided url."""
        self.url = os.path.join(url_base, url_extension)

    def get_url(self):
        return self.url

    def read_csv(self) -> pd.DataFrame:
        data = pd.read_csv(self.url)



