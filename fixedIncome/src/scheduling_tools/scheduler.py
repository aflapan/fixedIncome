"""
An object for
"""


from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd  # type: ignore
from fixedIncome.src.scheduling_tools.schedule_enumerations import Weekdays

from fixedIncome.src.scheduling_tools.holidays import (generate_all_holidays,
                                                       Holiday,
                                                       get_next_weekday)

class Scheduler(object):

    @staticmethod
    def get_next_weekday(date_obj: date, weekday: Weekdays) -> date:
        return get_next_weekday(date_obj, weekday)

    # ---------------------------------------------------------------
    # Business Day Increment Functionality

    @staticmethod
    def add_business_days(date_obj: date, business_days: int, holiday_calendar: dict[str, Holiday]) -> date:
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
        while not Scheduler.is_business_day(date_obj, holiday_calendar):  # while not a business day, increment
            date_obj += timedelta(days=1)                                 # generates the starting business day.

        increment = 1 if business_days >= 0 else -1

        for _ in range(0, business_days, increment):
            date_obj = Scheduler.add_single_business_day(date_obj, holiday_calendar) if business_days >= 0 \
                else Scheduler.subtract_single_business_day(date_obj, holiday_calendar)

        return date_obj

    @staticmethod
    def add_single_business_day(date_obj: date, holiday_calendar: dict[str, Holiday]) -> date:
        """
        Adds a single Business Day. Assumes that date is a valid New York Business Day, and will
        raise an exception otherwise.

        Examples include:
        date(2022, 12, 21) -> date(2022, 12, 22) as both dates are New York Business Days
        date(2022, 12, 23) -> date(2022, 12, 27) as the 24th and 25th are weekend days and the 26 is adjusted Christmas
        """

        if not Scheduler.is_business_day(date_obj, holiday_calendar):
            raise ValueError(f"{date_obj} is not a New York Business day.")

        date_obj += timedelta(days=1)  # add the single day

        while not Scheduler.is_business_day(date_obj, holiday_calendar):  # Keep adding days until result
                                                                      # is a new york business day
            date_obj += timedelta(days=1)

        return date_obj

    @staticmethod
    def subtract_single_business_day(date_obj: date, holiday_calendar: dict[str, Holiday]) -> date:
        """
        Subtracts a single business day. If the provided date object is not a business day, then
        it will return the last business day before the provided date.

        Examples Include:
        date(2022, 12, 22) -> date(2022, 12, 21) as both dates are New York Business Days
        date(2022, 12, 27) -> date(2022, 12, 23) as the 24th and 25th are weekend days and the 26 is adjusted Christmas
        """

        if not Scheduler.is_business_day(date_obj, holiday_calendar):
            date_obj = Scheduler.add_business_days(date_obj, business_days=0, holiday_calendar=holiday_calendar)
        date_obj += timedelta(days=-1)  # subtract the single day

        while not Scheduler.is_business_day(date_obj, holiday_calendar):
            date_obj += timedelta(days=-1)  # keep subtracting until days

        return date_obj

    @staticmethod
    def is_business_day(date_obj: date, holiday_calendar: dict[str, Holiday]) -> bool:
        """
        Tests whether the provided date is a New York Business Day.
        """
        is_weekday = date_obj.weekday() != Weekdays.SATURDAY.value and date_obj.weekday() != Weekdays.SUNDAY.value
        year = date_obj.year
        not_holiday = date_obj not in generate_all_holidays(from_year=year,
                                                            to_year=year,
                                                            holiday_calendar=holiday_calendar)

        return not_holiday and is_weekday

    @staticmethod
    def generate_us_business_days(start_date: date, end_date: date, holiday_calendar: dict) -> list[date]:
        """
        Generates a list of all us business days which lie (inclusively) between
        the provided from_date and to_date.
        """
        busisness_days = []
        current_day = Scheduler.add_business_days(start_date, business_days=0, holiday_calendar=holiday_calendar)
        while current_day <= end_date:
            busisness_days.append(current_day)
            current_day = Scheduler.add_single_business_day(current_day, holiday_calendar=holiday_calendar)
        return busisness_days

    @staticmethod
    def generate_dates_by_increments(
            start_date: date, end_date: date, increment: timedelta | relativedelta, max_dates: int = 1_000
    ) -> list[date]:
        """
        Generates a sequence of dates starting at the start date, incrementing by the provided increment,
        and ending on either the end_date or the first date on the increment date which occurs before the end_date.
        By providing a positive or negative increment, one may either generate increasing or decreasing sequences.
        The max_dates parameter protects the user from inadvertently providing an increment in the wrong direction
        of the start_date -> end_date interval and entering an infinite loop and depleting memory.
        """
        count = 0
        dates = []
        date_obj = start_date
        while date_obj <= end_date and count <= max_dates:
            dates.append(date_obj)
            date_obj += increment
            count += 1

        return dates


    @staticmethod
    def following_date_adjustment(date_obj: date, holiday_calendar: dict) -> date:
        """
        Method for performing the following business day convention adjustment. If the provided
        date is a business day, then returns the original date. Otherwise, returns the
        first business day immediately following the provided date.
        """

        return Scheduler.add_business_days(date_obj, business_days=0, holiday_calendar=holiday_calendar)  # T+0 adjustment

    @staticmethod
    def modified_following_date_adjustment(date_obj: date, holiday_calendar: dict) -> date:
        """
        Method for performing the modified-following business day convention adjustment.
        Performs the following business day adjustment if the adjusted date is in the same month. Otherwise,
        the adjusted date becomes the previous business day.

        Reference:
        https://www.nasdaq.com/glossary/m/modified-following-businessday-convention
        """

        candidate_adjusted_date = Scheduler.add_business_days(date_obj, business_days=0, holiday_calendar=holiday_calendar)

        if candidate_adjusted_date.month == date.month:
            return candidate_adjusted_date

        else:
            return Scheduler.add_business_days(date_obj, business_days=-1, holiday_calendar=holiday_calendar)



