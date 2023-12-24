from datetime import date, datetime
from abc import abstractmethod
import math
from typing import Optional
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention, DayCountCalculator

class AffineModelMixin:
    """
    A Mixin class for including functionality for zero-coupon bond prices and yields
    based on an affine yield model.
    """
    bond_price_state_variable_coeffs: Optional[dict] = None
    yield_state_variable_coeffs: Optional[dict] = None

    @abstractmethod
    def _create_bond_price_coeffs(self, maturity_date: date) -> None:
        """
        Private helper method to set the coefficients beta_0, ..., beta_p of the state variables
        used to price zero coupon bonds using the formula:

        Zero Coupon Price = exp( beta_0 + beta_1 variable_1 + ... + beta_p variable_p )
        """

    @abstractmethod
    def _create_bond_yield_coeffs(self, maturity_date) -> None:
        """
        Private helper method to set the coefficients beta_0, ..., beta_p of the state variables
        used to calculate zero coupon bond yields:

        Zero Coupon Yield = beta_0 + beta_1 variable_1 + ... + beta_p variable_p
        """

    @abstractmethod
    def zero_coupon_bond_price(self, maturity_date: date, *args, **kwargs) -> float:
        """
        Calculates the price of a zero coupon bond.
        """

    @abstractmethod
    def zero_coupon_bond_yield(self, maturity_date: date, *args, **kwargs) -> float:
        """
        Calculates the time-t yield of a T-maturity bond, where t < T.
        The continuously-compounded yield is defined to be -log(P_t^T) / (T-t).
        """

    @abstractmethod
    def instantaneous_forward_rate(self, forward_rate_datetime: date | datetime) -> float:
        """
        Returns the time-t instantaneous forward rate based on the affine yield model parameters.
        """

