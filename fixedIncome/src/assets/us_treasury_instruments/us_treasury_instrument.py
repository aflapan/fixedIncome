from datetime import date
from typing import Optional
from abc import ABC, abstractmethod

from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention
from fixedIncome.src.scheduling_tools.schedule_enumerations import BusinessDayAdjustment
from fixedIncome.src.curves.curve import KnotValuePair
from fixedIncome.src.curves.yield_curves.yield_curve import YieldCurve
from fixedIncome.src.assets.cashflow import CashflowCollection

class UsTreasuryInstrument(ABC, CashflowCollection):
    """ A base class which defines the necessary attributes and methods
    for US Treasury Instruments (i.e. Bonds and Futures). """

    def __init__(self,
                 market_quote: float,
                 principal: float,
                 purchase_date: date,
                 settlement_date: date,
                 maturity_date: date,
                 day_count_convention: DayCountConvention,
                 business_day_adjustment: BusinessDayAdjustment,
                 cusip: Optional[str] = None
                 ) -> None:

        self.market_quote = market_quote
        self.principal = principal
        self.purchase_date = purchase_date
        self.settlement_date = settlement_date
        self.maturity_date = maturity_date
        self.day_count_convention = day_count_convention
        self.business_day_adjustment = business_day_adjustment
        self.cusip = cusip

    @abstractmethod
    def present_value(self, curve: YieldCurve) -> float:
        return NotImplemented('Please provide an implementation for present_value.')

    @abstractmethod
    def to_knot_value_pair(self) -> KnotValuePair:
        """ Returns a KnotValuePair """
        return NotImplemented('Please provide an implementation for _to_knot_value_pair.')



