"""
This script contains the unit tests for fixedIncome.src.scheduling_tools.holidays.py
"""
from datetime import date
from fixedIncome.src.scheduling_tools.schedule_enumerations import Weekdays
from fixedIncome.src.scheduling_tools.holidays import (US_FEDERAL_HOLIDAYS,
                                                       us_federal_holiday_adjustment,
                                                       get_next_weekday,
                                                       generate_holiday_dates,
                                                       generate_all_holidays)


def test_christmas_day_2022_is_adjusted_to_following_monday() -> None:
    """ Christmas day 2022-12-25 falls on a Sunday. The US federal holiday
    adjustment rules dictate that the holiday should be celebrated on the
    following Monday. This test ensures that the correct adjustment is made.
    """
    Christmas_day = date(2022, 12, 25)
    adjusted_Christmas_day = us_federal_holiday_adjustment(Christmas_day)
    assert (Christmas_day.weekday() == Weekdays.SUNDAY.value) \
           and (adjusted_Christmas_day.weekday() == Weekdays.MONDAY.value)


def test_generate_holidays_for_Christmas() -> None:
    Christmas_set = generate_holiday_dates('Christmas',
                                           from_year=2022,
                                           to_year=2023,
                                           holiday_calendar=US_FEDERAL_HOLIDAYS)

    assert Christmas_set == {date(2022, 12, 26), date(2023, 12, 25)}

def test_all_holidays_are_correctly_adjusted_for_2023_us_federal_holidays() -> None:
    """
    Tests that the US Federal Holidays are all correctly Adjusted given for the year 2023.
    """

    test_dates = {
        date(2023, 1, 2),    # adjust New Year's
        date(2023, 1, 17),   # MLK day
        date(2023, 2, 21),   # President's Day
        date(2023, 5, 30),   # Memorial Day
        date(2023, 6, 19),   # Juneteenth
        date(2023, 7, 4),    # Fourth of July / Independence Day
        date(2023, 9, 4),    # Labor Day
        date(2023, 10, 9),   # Columbus Day
        date(2023, 11, 10),  # Veterans' Day
        date(2023, 11, 23),  # Thanksgiving
        date(2023, 12, 25)   # Christmas
    }

    generated_holidays = generate_all_holidays(from_year=2023, to_year=2023, holiday_calendar=US_FEDERAL_HOLIDAYS)
    assert test_dates == generated_holidays

def test_get_next_Friday() -> None:
    """
    Tests that the function to obtain the following weekday produces the expected results.
    Specific test for getting the following Friday.
    """
    start_date = date(2023, 10, 28)  # a Saturday
    following_Friday = date(2023, 11, 3)
    assert get_next_weekday(start_date, Weekdays.FRIDAY) == following_Friday

def test_get_next_Saturday() -> None:
    """
    Tests that the function to obtain the following weekday produces the expected results.
    Specific test for getting the next Saturday, which would be the start date.
    """
    start_date = date(2023, 10, 28)  # a Saturday
    assert get_next_weekday(start_date, Weekdays.SATURDAY) == start_date

