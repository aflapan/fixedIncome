
import urllib.request
from typing import Optional
from datetime import date, datetime



class WebScraper:

    def __init__(self, base_url="https://www.treasurydirect.gov/TA_WS/securities/"):
        """

        Example:
            https://www.treasurydirect.gov/TA_WS/securities/Bill
        """
        self.base_url = base_url
        self.types = {'Bill', 'Note', 'Bond', 'CMB', 'TIPS', 'FRN'}

        self.field_parsers = {'issueDate': WebScraper.parse_datetime,
                              'securityTerm': WebScraper.parse_term,
                              'security_type': type_to_payment_freq,
                              'maturityDate': WebScraper.parse_datetime,
                              'interestRate': WebScraper.parse_float,
                              'accruedInterestPer100': WebScraper.parse_float,
                              'adjustedPrice': WebScraper.parse_float,
                              }

    def get_url(self, security_type: str) -> str:
        if security_type in self.types:
            url = self.base_url + security_type
            return url

        else:
            raise ValueError(f'{security_type} not found in security types. Types are ' + ', '.join(list(self.types)))



    def read_url_into_html_str(self, url: str) -> str:
        web_page = urllib.request.urlopen(url)
        page_in_html_bytes = web_page.read()
        html_str = page_in_html_bytes.decode("utf-8")
        return html_str

    def post_process_html_str(self, html_str) -> list[list[str]]:
        """

        """

        drop_chars = ['{', '[', ']', '}']
        for char in drop_chars:
            html_str = html_str.replace(char, '')

        cusip_list = html_str.split('cusip')[1:]  # index 0 corresponds to leading quote marks before 'cusip' string
        cusips = [cusip_info.split(',') for cusip_info in cusip_list][1:]  # 0-th index entry is leading commas
        return cusips

    #-----------------------------------------------------------------------------
    @staticmethod
    def parse_term(term_str: str) -> str:
        num_str, unit_str = term_str.split('-')
        match unit_str:
            case 'Week':
                return num_str+'W'
            case 'Month':
                return num_str + 'M'
            case _:
                raise ValueError(f'Time unit {unit_str} not recognized.')

    @staticmethod
    def parse_datetime(datetime_str: str) -> Optional[date]:
        if datetime_str == '':
            return None
        (date_str, time_str) = datetime_str.split('T')
        parsed_date = datetime.strptime(date_str, '%Y-%M-%D')
        return parsed_date

    @staticmethod
    def parse_float(str_val) -> Optional[float]:
        if str_val == '':
            return None
        return float(str_val)

    def type_to_payment_freq(self, type_str) -> Optional[float]:
        if type_str not in self.types:
            return None











if __name__ == '__main__':
    web_scraper = WebScraper()

    tbill_url = web_scraper.get_url('Bill')
    html_str = web_scraper.read_url_into_html_str(tbill_url)



