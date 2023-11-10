from datetime import date, datetime
from abc import abstractmethod
import math
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention, DayCountCalculator

class AffineModelMixin:
    """
    A Mixin class for including functionality for zero-coupon bond prices and yields
    based on an affine yield model.
    """

    @abstractmethod
    def time_t_zero_coupon_bond_price(self, price_date: date, maturity_date: date) -> float:
        """
        Calculates the price of a zero coupon bond.
        """

    def time_t_yield(self,
                     yield_date: date,
                     maturity_date: date,
                     day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_OVER_ACTUAL) -> float:
        """
        Calculates the time-t yield of a T-maturity bond, where t < T.
        The continuously-compounded yield is defined to be -log(P_t^T) / (T-t).
        """
        time_t_price = self.time_t_zero_coupon_bond_price(price_date=yield_date, maturity_date=maturity_date)
        accrual = DayCountCalculator.compute_accrual_length(yield_date, maturity_date, day_count_convention)
        return -math.log(time_t_price) / accrual

    @abstractmethod
    def time_t_forward_rate(self, forward_rate_datetime: date | datetime) -> float:
        """
        Returns the time-t instantaneous forward rate based on the affine yield model parameters.
        """




