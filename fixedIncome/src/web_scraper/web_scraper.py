
import urllib.request
from typing import Optional, Iterable, Any, Callable
from datetime import date, datetime
from dataclasses import dataclass
import itertools
from collections import defaultdict

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

    def __init__(self,
                 base_url: str = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service",
                 endpoint: str = "/v2/accounting/od/avg_interest_rates"):
        """
        The default endpoint provided is the average interest rates on U.S. Treasury Securities.

        Reference: https://fiscaldata.treasury.gov/api-documentation/
        """
        self.base_url = base_url
        self.endpoint = endpoint
        self.full_url = base_url + '/' + endpoint

        self.reference_date = date.today()

        self.field_parsers: defaultdict[Any, Callable[[str], Any]]
        self.field_parsers = defaultdict(lambda: None)
        self.field_parsers.update(**{
            'cusip': lambda val: str(val),
            'issueDate': WebScraper.parse_datetime,
            'securityTerm': WebScraper.parse_term,
            'securityType': WebScraper.asset_type_to_payment_freq,
            'maturityDate': WebScraper.parse_datetime,
            'interestRate': WebScraper.parse_float,
            'accruedInterestPer100': WebScraper.parse_float,
            'adjustedPrice': WebScraper.parse_float,
            'unadjustedPrice': WebScraper.parse_float,
            'datedDate': WebScraper.parse_datetime,
        })

        self.filter_fxcns = [self.maturity_date_is_future,
                             self.interest_rate_is_not_None]


    def read_url_into_html_str(self, url: Optional[str] = None) -> str:
        """
        Reads web download from the provided url and decodes the html bytes
        using utf-8. Returns the string for downstream processing.
        """
        if url is None:
            url = self.full_url

        url_reader = urllib.request.urlopen(url)
        page_in_html_bytes = url_reader.read()
        html_str = page_in_html_bytes.decode("utf-8")
        #html_str = html_str.lstrip('[').strip(']')  # remove leading and ending brackets to get cusips bracketed by { }
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
                parser_fxcn = self.field_parsers[key]
                if parser_fxcn is not None:
                    cusip_dict[key] = parser_fxcn(val)

            cusip_list.append(Cusip(**cusip_dict))

        return cusip_list

    #-------------------------------------------------------------------------
    # Filtering functionality

    def unadjustPrice_is_not_None(self, cusip: Cusip) -> bool:
        "Returns a boolean signifying whether the cusip's unadjusted price is not None."
        return cusip.unadjustedPrice is not None

    def maturity_date_is_future(self, cusip: Cusip) -> bool:
        "Returns a boolean signifying whether the cusip's maturity date is in the future."
        return cusip.maturityDate >= self.reference_date

    def interest_rate_is_not_None(self, cusip: Cusip) -> bool:

        return cusip.interestRate is not None


    def all_filters(self, cusip: Cusip) -> bool:
        """
        Returns a boolean signifying whether the cusip's
        passes all the tests in self.filter_fxcns.
        """
        return all(filter_fxcn(cusip) for filter_fxcn in self.filter_fxcns)


    def filter_cusips(self, cusips: Iterable[Cusip]) -> list[Cusip]:
        """
        Returns a list of cusips which passes all filters.
        """
        return list(filter(self.all_filters, cusips))




    #-----------------------------------------------------------------------------
    # Helper functions for parsing terms in cusip strings

    @staticmethod
    def parse_term(term_str: str) -> str:
        """ conjoins the individual date parsed terms """
        return '-'.join(WebScraper.parse_individual_term(part) for part in term_str.split())

    @staticmethod
    def parse_individual_term(term_str: str) -> str:
        """ Parses an individual string term """
        num_str, unit_str = term_str.split('-')
        match unit_str:
            case 'Week':
                return num_str+'W'
            case 'Month':
                return num_str + 'M'
            case 'Year':
                return num_str+'Y'
            case 'Day':
                return num_str+'D'
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
    html_str = web_scraper.read_url_into_html_str()





