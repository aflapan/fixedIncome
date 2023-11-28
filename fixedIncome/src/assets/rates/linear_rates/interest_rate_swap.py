
from datetime import date
import pandas as pd
from typing import Optional

from fixedIncome.src.scheduling_tools.holidays import Holiday
from fixedIncome.src.scheduling_tools.schedule_enumerations import BusinessDayAdjustment, SettlementConvention

class InterestRateSwap:
    def __init__(self,
                 float_index,
                 quote: float,
                 maturity_date: date,
                 purchase_date: date,
                 day_count_convention: str,
                 holiday_calendar: dict[str, Holiday],
                 settlement_convention: SettlementConvention = SettlementConvention.T_PLUS_TWO_BUSINESS,
                 business_day_adjustment: BusinessDayAdjustment = BusinessDayAdjustment.MODIFIED_FOLLOWING) -> None:
        self.float_index = float_index
        self.quote = quote
        self.maturity_date = maturity_date
        self.purchase_date = purchase_date
        self.day_count_convention = day_count_convention
        self.implied_quote = None



    @property
    def floating_leg(self) -> pd.DataFrame:
        return self._floating_leg

    @property
    def fixed_leg(self) -> pd.DataFrame:
        return self._fixed_leg

    def generate_floating_leg_cashflow_schedule(self) -> None:
        pass

    def generate_fixed_leg_cashflow_schedule(self) -> None:
        pass
