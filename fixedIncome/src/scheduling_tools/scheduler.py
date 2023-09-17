import datetime
from dateutil.relativedelta import relativedelta
import collections
import pandas as pd  # type: ignore
from fixedIncome.src.scheduling_tools.schedule_enumerations import PaymentFrequency, BusinessDayAdjustment

import enum



class Scheduler(object):
    def __init__(self, tenor: str,
                 purchase_date: datetime.date,
                 maturity_date: datetime.date,
                 settlement_convention: str,
                 payment_frequency: PaymentFrequency,
                 business_day_adjustment: BusinessDayAdjustment) -> None:
        """

        :param purchase_date:

        :param maturity_date:

        :param settlement_convention:
            Valid strings are 'T+0', 'T+1', and 'T+2'.
        :param payment_frequency:
            Valid strings are 'quarterly', 'semi-annual', 'annual'
        :param business_day_adjustment:
            Valid strings are 'following', 'modified following'.
        """
        self.tenor = tenor
        self.purchase_date = purchase_date
        self.maturity_date = maturity_date
        self.settlement_convention = settlement_convention
        self.payment_frequency = payment_frequency
        self.business_day_adjustment = business_day_adjustment

        self.weekday_enumeration = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')

        # Define holidays with corresponding adjustments
        Holiday = collections.namedtuple("Holiday", ['month', 'day', 'adjustment_fxcn'])

        self.us_federal_holiday_date_templates = {
            'NewYears': Holiday(1, 1, self._us_federal_holiday_adjustment),                          # Jan 1st
            'MLK': Holiday(1, 17, self._us_federal_holiday_adjustment),                              # Jan 17th
            'Presidents': Holiday(2, 21, self._us_federal_holiday_adjustment),                       # Feb 21st
            'Memorial': Holiday(5, 30, self._us_federal_holiday_adjustment),                         # May 30th
            'Juneteenth': Holiday(6, 19, self._us_federal_holiday_adjustment),                       # June 19th
            'Independence': Holiday(7, 4, self._us_federal_holiday_adjustment),                      # July 4th
            'Labor': Holiday(9, 1, lambda date: self._get_next_weekday(date, 'Monday')),             # first Monday in Sept
            'Columbus': Holiday(10, 8, lambda date: self._get_next_weekday(date, 'Monday')),         # second Monday in Oct
            'Veterans': Holiday(11, 11, self._us_federal_holiday_adjustment),                        # Nov 11th
            'Thanksgiving': Holiday(11, 22, lambda date: self._get_next_weekday(date, 'Thursday')),  # fourth Thursday of Nov
            'Christmas': Holiday(12, 25, self._us_federal_holiday_adjustment)                        # Dec 25th
        }

        self.new_york_business_holidays = self._generate_new_york_business_holidays()

        self.settlement_date = self._calculate_settlement_date()

        self.dated_date = self._calc_dated_date()

        self.payment_schedule = self.create_payment_schedule()

    #-----------------------------------------------------------------------
    # Payment Schedule Functionality

    def _calc_dated_date(self) -> datetime.date:
        """
        Calculates the dated date, the date on which interest begins accruing. Does this by parsing the
        Assumes self.tenor is of the form {int}Y or {int}M for a number of years or months, respectively.
        """
        length_str_slice = slice(-1)
        length = int(self.tenor[length_str_slice])

        match(self.tenor[-1]):
            case 'Y':
                dated_date = self.maturity_date - relativedelta(years=length)

            case 'M':
                dated_date = self.maturity_date - relativedelta(months=length)

            case _:
                raise ValueError(f'{self.tenor[-1]} is not a valid tenor unit indicator. '
                                 f'Tenor must be in the form [int]Y or [int]M.')

        return dated_date


    def create_payment_schedule(self) -> pd.DataFrame:
        """
        Creates a pandas dataframe of the payment schedule. Includes the maturity date, all coupon
        payments after the purchase date, and one coupon payment before the purchase date (if available)
        in order to calculate accrued interest. The dated date (beginning of accrued interest),
        purchase date, and settlement date

                  Date       Date Type Adjusted Date
        0   2023-06-15  coupon payment    2023-06-15
        1   2023-12-15  coupon payment    2023-12-15
        2   2024-06-15  coupon payment    2024-06-17
        3   2024-12-15  coupon payment    2024-12-16
        4   2025-06-15  coupon payment    2025-06-16
            ...               ...            ...
        37  2041-12-15  coupon payment    2041-12-16
        38  2042-06-15  coupon payment    2042-06-16
        39  2042-12-15  coupon payment    2042-12-15
        40  2043-06-15  coupon payment    2043-06-15
        41  2043-12-15  coupon payment    2043-12-15
        """

        # first step is to generate the payment days without holiday adjustments

        match self.payment_frequency:
            case PaymentFrequency.ZERO_COUPON:
                increment = None

            case PaymentFrequency.QUARTERLY:
                increment = relativedelta(months=3)

            case PaymentFrequency.SEMI_ANNUAL:
                increment = relativedelta(months=6)

            case PaymentFrequency.ANNUAL:
                increment = relativedelta(years=1)

            case _:
                raise ValueError(f"{self.payment_frequency} is not a valid payment frequency.")

        payment_dates = [(self.maturity_date, 'maturity date')]
        date = self.maturity_date

        if self.payment_frequency != 'zero-coupon':
            while date > self.dated_date + increment:
                date = date - increment
                payment_dates.append((date, 'coupon payment'))

        payment_dates.sort(key = lambda date_description_pair: date_description_pair[0]) # sort dates

        self.payment_schedule = pd.DataFrame(payment_dates, columns = ['Date', 'Date Type'])

        # Second step is to adjust the payment days based on the provided business_day_adjustment

        # date adjustment fxcn
        match self.business_day_adjustment:
            case BusinessDayAdjustment.FOLLOWING:
                adjsutment_fxcn = self._following_date_adjustment

            case BusinessDayAdjustment.MODIFIED_FOLLOWING:
                adjsutment_fxcn = self._modified_following_date_adjustment

            case _:
                raise ValueError(f" Business day adjustment {self.business_day_adjustment} is invalid.")

        self.payment_schedule['Adjusted Date'] = self.payment_schedule.apply(
            lambda row: row['Date'] if row['Date Type'] in ['dated date', 'purchase date'] else adjsutment_fxcn(row['Date']),
            axis=1
        )

        return self.payment_schedule


    def get_payment_schedule(self) -> pd.DataFrame:
        """
        Retrieves the pandas data frame of the payment schedule.
        """

        return self.payment_schedule


    def _calculate_settlement_date(self) -> datetime.date:
        """
        Method to compute the settlement date based on the purchase date and the settlement_convention.
        """
        match self.settlement_convention:
            case "T+0 business":
                return self.add_new_york_business_days(self.purchase_date, business_days=0)

            case "T+1 business":
                return self.add_new_york_business_days(self.purchase_date, business_days=1)

            case "T+2 business":
                return self.add_new_york_business_days(self.purchase_date, business_days=2)

            case _:
                raise ValueError(f"Settlement Convention {self.settlement_convention} is invalid. Valid strings are "
                                 f"'T+0 business', 'T+1 business', 'T+2 business'.")

    # --------------------------------------------------------------
    # Holiday Functionality

    def get_new_york_business_holidays(self) -> set:
        """
        Returns the set containing all us federal holidays starting from the purchase date year
        up to 100 years.
        """
        return self.new_york_business_holidays


    def _generate_new_york_business_holidays(self, years: int = 100) -> set:
        """
        Generates all New York Business Holidays starting at the self.purchase_date year
        and ending at the self.purchase_date year + the specified number of years (default 100).
        Holidays are generated using the templates and date adjustments found in the dict
        self.us_federal_holiday_date_templates.

        Reference:
        https://www.federalreserve.gov/aboutthefed/k8.htm
        """
        holidays = set()

        for holiday_name in self.us_federal_holiday_date_templates.keys():
            holidays |= self._generate_us_holiday_dates(holiday_name, years)

        return holidays


    def _generate_us_holiday_dates(self, holiday_name: str, years: int = 100) -> set:
        """
        Generates all federal holidays for a particular holiday given the holiday name and the number of years.
        Starts with the holiday in the purchase date year and ends with purchase date year + years.

        Adjusts all dates according to the adjustment functions provided in the named tuples inside
        self.us_federal_holiday_dates.
        """

        try:
            holiday_tuple = self.us_federal_holiday_date_templates[holiday_name]
        except KeyError:
            raise KeyError(f"{holiday_name} is not a valid holiday name. "
                           f"Valid holidays are {tuple(self.us_federal_holiday_date_templates.keys())}")

        holidays = {holiday_tuple.adjustment_fxcn(datetime.date(year, holiday_tuple.month, holiday_tuple.day))
                    for year in range(self.purchase_date.year, self.purchase_date.year + years)}

        return holidays

    # ---------------------------------------------------------------
    # Business Day Increment Functionality

    def add_new_york_business_days(self, date:datetime.date, business_days:int) -> datetime.date:
        """
        Returns the date which is the provided number of business days away from the specified date.

        This method first adjusts the provided date by moving it to the business day on or immediately following
        the provided date (T+0). It then adds a single day and makes the same adjustment to give T+1. Iterating this
        process business_day number of times yields T+business_days final date. Schematically, if business_days = n
        for some integer n:

        date ---->  business day on or immediately following date gives T+0
        (T+0 + 1 day) ----> business day on or immediately following gives T+1
        (T+1 + 1 day) ----> business day on or immediately following gives T+2
         .
         .
         .
         (T+n-1 + 1 day) ----> business day on or immediately following gives T+n
        """

        while not self._is_new_york_business_day(date): # while not a business day, increment
            date += datetime.timedelta(days = 1)        # generates the starting business day.

        increment = 1 if business_days >= 0 else -1

        for _ in range(0, business_days, increment):
            date = self._add_single_new_york_business_day(date) if business_days >= 0 \
                else self._subtract_single_new_york_business_day(date)

        return date


    def _add_single_new_york_business_day(self, date:datetime.date) -> datetime.date:
        """
        Adds a single New York Business Day. Assumes that date is a valid New York Business Day, and will
        raise an exception otherwise.

        Examples include:
        date(2022, 12, 21) -> date(2022, 12, 22) as both dates are New York Business Days
        date(2022, 12, 23) -> date(2022, 12, 27) as the 24th and 25th are weekend days and the 26 is adjusted Christmas
        """

        if not self._is_new_york_business_day(date):
            raise ValueError(f"{date} is not a New York Business day.")

        date += datetime.timedelta(days=1) # add the single day

        while not self._is_new_york_business_day(date): # Keep adding days until result is a new york business day
            date += datetime.timedelta(days=1)

        return date


    def _subtract_single_new_york_business_day(self, date:datetime.date) -> datetime.date:
        """
        Subtracts a single New York Business Day. Assumes that date is a valid New York Business Day, and will
        raise an exception otherwise.

        Examples Include:
        date(2022, 12, 22) -> date(2022, 12, 21) as both dates are New York Business Days
        date(2022, 12, 27) -> date(2022, 12, 23) as the 24th and 25th are weekend days and the 26 is adjusted Christmas
        """

        if not self._is_new_york_business_day(date):
            raise ValueError(f"{date} is not a New York Business day.")

        date += datetime.timedelta(days = -1) # subtract the single day

        while not self._is_new_york_business_day(date):
            date += datetime.timedelta(days = -1) # keep subtracting until days

        return date


    def _is_new_york_business_day(self, date:datetime.date) -> bool:
        """
        Tests whether the provided date is a New York Business Day.
        """

        is_weekday = self.weekday_enumeration[date.weekday()] not in ['Saturday', 'Sunday']
        return (date not in self.new_york_business_holidays) and is_weekday


    # ---------------------------------------------------------------
    # Date adjustment functions

    def _get_next_weekday(self, date: datetime.date, weekday: str) -> datetime.date:
        """
        Method to find the first specified weekday on or following the provided date.
        If the date is the provided weekday, returns the date. Otherwise, returns
        the date corresponding to the weekday immediately following the date.
        """
        if self.weekday_enumeration[date.weekday()] == weekday:
            next_weekday = date

        else:
            weekday_index = self.weekday_enumeration.index(weekday)
            next_weekday = date + datetime.timedelta(days= (weekday_index - date.weekday()) % 7)

        return next_weekday

    def _us_federal_holiday_adjustment(self, date: datetime.date) -> datetime.date:
        """
        Adjusts the day according to the US Federal Holiday Rules:
        1. If a holiday falls on a weekday, it stays as is.
        2. If a holiday falls on a Saturday, adjust it to the preceeding Friday.
        3. If a holiday falls on a Sunday, adjust it to the following Monday.

        Reference:
        https://www.opm.gov/policy-data-oversight/pay-leave/work-schedules/fact-sheets/Federal-Holidays-In-Lieu-Of-Determination
        FAQ Question 1
        """
        match self.weekday_enumeration[date.weekday()]:
            case 'Saturday':
                return date + datetime.timedelta(days=-1)

            case 'Sunday':
                return date + datetime.timedelta(days=1)

            case _:
                return date


    def _following_date_adjustment(self, date:datetime.date) -> datetime.date:
        """
        Method for performing the following business day convention adjustment. If the provided
        date is a business day, then returns the original date. Otherwise, returns the
        first business day immediately following the provided date.
        """

        return self.add_new_york_business_days(date, business_days=0) # performs T+0 adjustment

    def _modified_following_date_adjustment(self, date:datetime.date) -> datetime.date:
        """
        Method for performing the modified-following business day convention adjustment.
        Performs the following business day adjustment if the adjusted date is in the same month. Otherwise,
        the adjusted date becomes the previous business day.

        Reference:
        https://www.nasdaq.com/glossary/m/modified-following-businessday-convention
        """

        candidate_adjusted_date = self.add_new_york_business_days(date, business_days=0)

        if candidate_adjusted_date.month == date.month:
            return candidate_adjusted_date

        else:
            return self.add_new_york_business_days(date, business_days=-1)

