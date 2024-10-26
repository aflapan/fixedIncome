"""
This script contains the collection of unit tests for
src/assets/rates/linear_rates/interest_rate_future.py
"""
from datetime import date

from fixedIncome.src.assets.rates.linear_rates.interest_rate_future import OneMonthSofrFuture
from fixedIncome.src.scheduling_tools.schedule_enumerations import Months
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.holidays import US_FEDERAL_HOLIDAYS


