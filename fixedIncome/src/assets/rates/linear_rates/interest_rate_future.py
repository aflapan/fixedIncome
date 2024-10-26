"""
This script implements Interest Rate Futures.

Unit tests contained in tests/test_assets/test_rates/test_linear_rates/test_interest_rate_future.py
"""

from typing import Callable
from datetime import date
from enum import Enum
import itertools


from fixedIncome.src.scheduling_tools.holidays import Holiday
from fixedIncome.src.assets.base_cashflow import CashflowCollection
from fixedIncome.src.scheduling_tools.schedule_enumerations import Months
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.curves.base_curve import Curve, DiscountCurve, KnotValuePair

class InterestRateFutureDirection(Enum):
    PAYER_FIXED = 0
    RECEIVER_FIXED = 1

IMMCodeToMonth = {
    'G': Months.FEBRUARY,
}


class InterestRateFuture(CashflowCollection):
    pass



class OneMonthSofrFuture(InterestRateFuture):
    def __init__(self,
                 contract_name: str,
                 purchase_date: date,
                 delivery_month: Months,
                 delivery_year: int,
                 price: float,
                 holiday_calendar: dict[str, Holiday],
                 notional: int = 5_000_000,
                 accrual_day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_360
                 ) -> None:
        self._contract_name = contract_name
        self._price = price
        self._purchase_date = purchase_date
        self._delivery_month = delivery_month
        self._delivery_year = delivery_year
        self._holiday_calendar = holiday_calendar
        self._notional = notional
        self._accrual_day_count_convention = accrual_day_count_convention

        self.last_trading_day = None

    @property
    def contract_name(self) -> str:
        return self._contract_name
    @property
    def price(self) -> float:
        return self._price

    @property
    def delivery_month(self) -> Months:
        return self._delivery_month

    @property
    def delivery_year(self) -> int:
        return self._delivery_year

    @property
    def purchase_date(self) -> date:
        return self._purchase_date

    @property
    def holiday_calendar(self) -> dict[str, Holiday]:
        return self._holiday_calendar

    @property
    def accrual_day_count_convention(self) -> DayCountConvention:
        return self._accrual_day_count_convention


    def present_value(self, discount_curve: DiscountCurve) -> float:
        """
        """

    def to_knot_value_pair(self) -> KnotValuePair:
        """
        """

    def get_last_trading_date(self) -> date:
        """
        The last trading date of the

        Reference: Tuckman and Serrat, 4th ed. Chapter ...
        """
        pass


    def calculate_settlement_rate(self, interest_rate: Callable[[date], float]) -> float:
        """
        Calculates the settlement rate
        """
        first_day_of_month = date()
        last_day_of_month = date()
        month_business_days = Scheduler.generate_business_days(start_date=first_day_of_month,
                                                               end_date=last_day_of_month,
                                                               holiday_calendar=self.holiday_calendar)

        weighted_sum_of_rates, total_accrual = 0.0, 0.0
        for start_date, end_date in itertools.pairwise(month_business_days):

            interest_rate_fixing = interest_rate(start_date)
            accrual = DayCountCalculator.compute_accrual_length(start_date=start_date,
                                                                end_date=end_date,
                                                                dcc=self.accrual_day_count_convention)
            weighted_sum_of_rates += interest_rate_fixing * accrual
            total_accrual += accrual

        return weighted_sum_of_rates/total_accrual









class ThreeMonthSofrFuture(InterestRateFuture):
    pass



