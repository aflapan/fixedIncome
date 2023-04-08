
import urllib.request
from typing import Optional
from datetime import date, datetime
from dataclasses import dataclass

@dataclass
class Cusip:
    cusip: Optional[str]
    issueDate: Optional[date]
    securityType: Optional[str]
    securityTerm: Optional[str]
    maturityDate: Optional[date]
    interestRate: Optional[float]
    datedDate: Optional[date]
    unadjustedPrice: Optional[float]
    adjustedPrice: Optional[float]
    accruedInterestPer100: Optional[float]


class WebScraper:

    def __init__(self, base_url="https://www.treasurydirect.gov/TA_WS/securities/"):
        """

        Example:
            https://www.treasurydirect.gov/TA_WS/securities/Bill
        """
        self.base_url = base_url
        self.types = {'Bill', 'Note', 'Bond', 'CMB', 'TIPS', 'FRN'}

        self.field_parsers = {
            'cusip': lambda val: str(val),  # enforces type
            'issueDate': WebScraper.parse_datetime,
            'securityTerm': WebScraper.parse_term,
            'securityType': WebScraper.asset_type_to_payment_freq,
            'maturityDate': WebScraper.parse_datetime,
            'interestRate': WebScraper.parse_float,
            'accruedInterestPer100': WebScraper.parse_float,
            'adjustedPrice': WebScraper.parse_float,
            'unadjustedPrice': WebScraper.parse_float,
            'datedDate': WebScraper.parse_datetime,
        }

    def get_url(self, security_type: str) -> str:
        if security_type in self.types:
            url = self.base_url + security_type
            return url

        else:
            raise ValueError(f'{security_type} not found in security types. Types are ' + ', '.join(list(self.types)))


    def read_url_into_html_str(self, url: str) -> str:
        url_reader = urllib.request.urlopen(url)
        page_in_html_bytes = url_reader.read()
        html_str = page_in_html_bytes.decode("utf-8")
        html_str = html_str.lstrip('[').strip(']')  # remove leading and ending brackets to get cusips bracketed by { }
        return html_str


    def parse_html_str_into_cusip_list(self, html_str) -> list[Cusip]:
        """

        """
        html_str = html_str.lstrip('{').rstrip('}')
        cusip_str_list = html_str.split('},{')
        cusip_list = []

        for cusip_str in cusip_str_list:

            # Tokenize cusip_str into a dictionary
            split_list = cusip_str.split(',')
            cusip_dict = dict()

            for key_val_str in split_list:
                seperator_index = key_val_str.find(':')
                key = key_val_str[:seperator_index].lstrip('"').rstrip('"')
                val = key_val_str[seperator_index+1:].lstrip('"').rstrip('"')

                parser_fxcn = self.field_parsers.get(key, None)
                if parser_fxcn is not None:
                    cusip_dict[key] = parser_fxcn(val)

            cusip_list.append(Cusip(**cusip_dict))

        return cusip_list

    def filter_cusip_list(self, cusips: list[Cusip]) -> list[Cusip]:
        """

        Returns a list of cusips which passes all of the filters.
        """
        [cusip for cusip in cusips if cusip.unadjustedPrice is not None]
        pass


    #-----------------------------------------------------------------------------
    @staticmethod
    def parse_term(term_str: str) -> str:
        num_str, unit_str = term_str.split('-')
        match unit_str:
            case 'Week':
                return num_str+'W'
            case 'Month':
                return num_str + 'M'
            case 'Year':
                return num_str+'Y'
            case _:
                raise ValueError(f'Time unit {unit_str} not recognized.')

    @staticmethod
    def parse_datetime(datetime_str: str) -> Optional[date]:
        if datetime_str == '':
            return None

        (date_str, time_str) = datetime_str.split('T')
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        return parsed_date

    @staticmethod
    def parse_float(str_val) -> Optional[float]:
        if str_val == '':
            return None
        return float(str_val)

    @staticmethod
    def asset_type_to_payment_freq(type_str) -> str:
        """
        Reference https://treasurydirect.gov/marketable-securities/[ASSET]
        """
        match type_str:
            case 'Bill':
                return 'zero-coupon'
            case 'Note':
                return 'semi-annual'
            case 'Bond':
                return 'semi-annual'
            case 'TIPS':
                return 'semi-annual'
            case 'CMB':
                return 'zero-coupon'
            case 'FRN':
                return 'quarterly'
            case _:
                raise ValueError(f'Asset type {type_str!r} is not a valid type.')









#--------------------------------------------------------------------------


if __name__ == '__main__':
    web_scraper = WebScraper()

    tbill_url = web_scraper.get_url('Bill')
    bill_html_str = web_scraper.read_url_into_html_str(tbill_url)
    bill_cusips = web_scraper.parse_html_str_into_cusip_list(bill_html_str)

    bond_url = web_scraper.get_url('Bond')
    bond_html_str = web_scraper.read_url_into_html_str(bond_url)
    bond_cusips = web_scraper.parse_html_str_into_cusip_list(bond_html_str)



