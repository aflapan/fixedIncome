"""
This script contains the set of Holidays templates and their corresponding adjustment functions.
An adjustment function takes the holiday and returns the date which is taken off. These templates
and their corresponding adjustment functions allow us to generate the specific holiday dates
for any given year.

Unit tests are found in fixedIncome.tests.test_scheduling_tools.test_holidays.py
"""

from datetime import date, timedelta
from typing import NamedTuple, Callable

from fixedIncome.src.scheduling_tools.schedule_enumerations import Months, Weekdays

# Define holiday template with corresponding adjustments
class Holiday(NamedTuple):
    Month: int | Months
    Day: int
    Adjustment_Function: Callable[[date], date]



def us_federal_holiday_adjustment(holiday_date: date) -> date:
    """
    Adjusts the day according to the US Federal Holiday Rules:
    1. If a holiday falls on a weekday, it stays as is.
    2. If a holiday falls on a Saturday, adjust it to the preceeding Friday.
    3. If a holiday falls on a Sunday, adjust it to the following Monday.

    Reference:
    https://www.opm.gov/policy-data-oversight/pay-leave/work-schedules/fact-sheets/Federal-Holidays-In-Lieu-Of-Determination
    FAQ Question 1
    """
    match holiday_date.weekday():
        case Weekdays.SATURDAY.value:
            return holiday_date + timedelta(days=-1)  # bump back to immediately preceeding Friday

        case Weekdays.SUNDAY.value:
            return holiday_date + timedelta(days=1)  # bump up to immediately following Monday

        case _:
            return holiday_date


def get_next_weekday(date_obj: date, weekday: Weekdays) -> date:
    """
    Method to find the first specified weekday on or following the provided date.
    If the date is the provided weekday, returns the date. Otherwise, returns
    the date corresponding to the weekday immediately following the date.

    For example, if we ask it to provide the next Thursday given some date, if the
    date is a Thursday, then the date itself is returned. Otherwise,
    the date of the Thursday immediately following the input date is returned.
    """
    if date_obj.weekday() == weekday.value:
        return date_obj

    else:
        days_diff = (weekday.value - date_obj.weekday()) % 7
        next_weekday = date_obj + timedelta(days=days_diff)
        return next_weekday



US_FEDERAL_HOLIDAYS = {
            'NewYears':     Holiday(Months.JANUARY, 1, us_federal_holiday_adjustment),                                     # Jan 1st
            'MLK':          Holiday(Months.JANUARY, 17, us_federal_holiday_adjustment),                                    # Jan 17th
            'Presidents':   Holiday(Months.FEBRUARY, 15, lambda date_obj: get_next_weekday(date_obj, Weekdays.MONDAY)),    # Third Monday of February
            'Memorial':     Holiday(Months.MAY, 30, us_federal_holiday_adjustment),                                        # May 30th
            'Juneteenth':   Holiday(Months.JUNE, 19, us_federal_holiday_adjustment),                                       # June 19th
            'Independence': Holiday(Months.JULY, 4, us_federal_holiday_adjustment),                                        # July 4th
            'Labor':        Holiday(Months.SEPTEMBER, 1, lambda date_obj: get_next_weekday(date_obj, Weekdays.MONDAY)),    # first Monday in Sept
            'Columbus':     Holiday(Months.OCTOBER, 8, lambda date_obj: get_next_weekday(date_obj, Weekdays.MONDAY)),      # second Monday in Oct
            'Veterans':     Holiday(Months.NOVEMBER, 11, us_federal_holiday_adjustment),                                   # Nov 11th
            'Thanksgiving': Holiday(Months.NOVEMBER, 22, lambda date_obj: get_next_weekday(date_obj, Weekdays.THURSDAY)),  # fourth Thursday of Nov
            'Christmas':    Holiday(Months.DECEMBER, 25, us_federal_holiday_adjustment)                                    # Dec 25th
        }


def generate_holiday_dates(
        holiday_name: str, from_year: int, to_year: int, holiday_calendar: dict[str, Holiday]
) -> set[date]:
        """
        Generates all holiday dates (with adjustments) for a particular holiday given the holiday name
        and the interval of years.

        Adjusts all dates according to the adjustment functions provided in the Holiday NamedTuple, indexed by
        the holiday_name string within the holiday_templates dictionary.
        """

        try:
            holiday_tuple = holiday_calendar[holiday_name]
        except KeyError:
            raise KeyError(f"{holiday_name} is not a valid holiday name. "
                           f"Valid holidays are {', '.join(holiday_calendar.keys())}.")

        month = holiday_tuple.Month.value if isinstance(holiday_tuple.Month, Months) else holiday_tuple.Month
        holidays = {holiday_tuple.Adjustment_Function(date(year, month, holiday_tuple.Day))
                    for year in range(from_year, to_year+1)}

        return holidays


def generate_all_holidays(from_year: int, to_year: int, holiday_calendar: dict[str, Holiday]) -> set[date]:
    """
    Generates all Holidays in the provided templated starting at the providing from_year
    up-to-and-including the to_year.
    Holidays are generated using the templates and date adjustments found in the provided holiday_templates.
    """
    holidays = set()

    for holiday_name in holiday_calendar.keys():
        holidays |= generate_holiday_dates(holiday_name, from_year, to_year, holiday_calendar)

    return holidays
