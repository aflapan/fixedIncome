"""
Unit tests for fixedIncome.src.scheduling_tools.scheduler.py
"""
from datetime import date
from dateutil.relativedelta import relativedelta
from fixedIncome.src.scheduling_tools.holidays import (US_FEDERAL_HOLIDAYS,
                                                       Holiday,
                                                       generate_all_holidays,
                                                       generate_holiday_dates)
from fixedIncome.src.scheduling_tools.scheduler import Scheduler

def test_add_business_day_weekday() -> None:
    """
    Tests that adding a single business day given a holiday calendar
    results in the correct date for a weekday (no Holiday).
    """
    # A standard weekday, adjusting 1 days
    start_date_weekday = date(2023, 10, 26)  # A Thursday, not a holiday
    next_business_day_weekday = date(2023, 10, 27)  # following Friday
    scheduled_next_business_day = Scheduler.add_business_days(start_date_weekday, 1, US_FEDERAL_HOLIDAYS)
    assert next_business_day_weekday == scheduled_next_business_day

def test_add_business_day_weekend() -> None:
    """
    Tests that adding business days given a holiday calendar results in the correct dates for a Friday.
    """
    # a standard weekday, adjusting 1 day over the weekend
    start_date_weekend = date(2023, 10, 27)  # Friday
    next_business_day_weekend = date(2023, 10, 30)  # following Monday
    scheduled_next_business_day = Scheduler.add_business_days(start_date_weekend, 1, US_FEDERAL_HOLIDAYS)
    assert next_business_day_weekend == scheduled_next_business_day

def test_adding_two_business_days_weekday() -> None:
    """
    Tests that adding two business days given a holiday calendar results in the correct dates for a Thursday.
    """
    # a standard weekday, adjusting 2 days over the weekend
    start_date_weekday = date(2023, 10, 26)  # A Thursday, not a holiday
    next_two_business_day_weekday = date(2023, 10, 30)   # following Friday
    scheduled_next_business_day = Scheduler.add_business_days(start_date_weekday, 2, US_FEDERAL_HOLIDAYS)
    assert next_two_business_day_weekday == scheduled_next_business_day

def test_adding_a_business_day_Christmas() -> None:
    """
    Tests that adding a business day given a holiday calendar results in the correct dates for the Friday before
    Christmas, 2023.
    """
    # Holiday and a weekend, adjusting 1 day
    start_date_holiday = date(2023, 12, 22)  # Friday before Christmas
    next_business_day_holiday = date(2023, 12, 26)  # Tuesday following Christmas
    scheduled_next_business_day = Scheduler.add_business_days(start_date_holiday, 1, US_FEDERAL_HOLIDAYS)
    assert next_business_day_holiday == scheduled_next_business_day


def test_subtracting_a_business_day_from_day_after_adjusted_Christmas() -> None:
    """
    Tests that subtracting a single business day works across Christmas holiday and weekend.
    """
    start_date = date(2022, 12, 27)  # Tuesday after Christmas, 2022
    expected_result = date(2022, 12, 23)  # The Friday before the weekend, and Christmas adjusted to Monday
    assert Scheduler.subtract_single_business_day(start_date, US_FEDERAL_HOLIDAYS) == expected_result


def test_subtracting_a_business_day_from_a_normal_business_Tuesday() -> None:
    """
    Tests that subtracting a single business day works for a standard Tuesday without weekends or holidays.
    """
    start_date = date(2023, 10, 24)  # A random Tuesday
    expected_result = date(2023, 10, 23)  # the previous Monday
    assert Scheduler.subtract_single_business_day(start_date, US_FEDERAL_HOLIDAYS) == expected_result


def test_generate_business_days_across_weekend() -> None:
    """
    Tests that the set of business days across two weeks, without a Holiday in between, contains
    Monday-Friday for those two weeks.
    """
    expected_result = [
        date(2023, 10, 16), date(2023, 10, 17), date(2023, 10, 18), date(2023, 10, 19), date(2023, 10, 20),
        date(2023, 10, 23), date(2023, 10, 24), date(2023, 10, 25), date(2023, 10, 26), date(2023, 10, 27)
     ]

    test_results = Scheduler.generate_business_days(start_date=date(2023, 10, 15),
                                                    end_date=date(2023, 10, 29),
                                                    holiday_calendar=US_FEDERAL_HOLIDAYS)
    assert test_results == expected_result

def test_generate_date_by_increments() -> None:
    """
    Tests that generating dates starting with a date, ending with a future date,
    and incrementing by a positive unit of time generates an increasing sequence of dates.
    """
    increment = relativedelta(months=1)
    start_date = date(2022, 1, 1)
    end_date = date(2023, 1, 1)
    expected_result = [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1), date(2022, 4, 1),
                       date(2022, 5, 1), date(2022, 6, 1), date(2022, 7, 1), date(2022, 8, 1),
                       date(2022, 9, 1), date(2022, 10, 1), date(2022, 11, 1), date(2022, 12, 1), date(2023, 1, 1)]
    test_result = Scheduler.generate_dates_by_increments(start_date, end_date, increment)

    assert test_result == expected_result


def test_generate_date_by_increments_negative_increments() -> None:
    """
    Tests that generating dates starting with a date in the future, ending with a previous date,
    and incrementing by a negative unit of time generates a decreasing sequence of dates.
    """
    increment = relativedelta(months=-1)
    start_date = date(2023, 1, 1)
    end_date = date(2022, 1, 1)
    expected_result = [date(2023, 1, 1), date(2022, 12, 1), date(2022, 11, 1), date(2022, 10, 1),
                       date(2022, 9, 1), date(2022, 8, 1), date(2022, 7, 1), date(2022, 6, 1),
                       date(2022, 5, 1), date(2022, 4, 1), date(2022, 3, 1), date(2022, 2, 1), date(2022, 1, 1)]
    test_result = Scheduler.generate_dates_by_increments(start_date, end_date, increment)

    assert test_result == expected_result


def test_modified_following_adjustment_goes_back_at_months_end() -> None:
    """
    tests that the modified following business day adjustment correctly goes backwards when the initial date
    is a non-business day at the end of the month.
    """
    start_date = date(2023, 12, 31)
    expected_result = date(2023, 12, 29)
    test_result = Scheduler.modified_following_date_adjustment(start_date, US_FEDERAL_HOLIDAYS)
    assert test_result == expected_result


