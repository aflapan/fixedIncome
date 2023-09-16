
from datetime import date
import pandas as pd
from typing import Optional

US_INDICES = {'SOFR', 'LIBOR-3M', 'FEDERAL FUNDS'}

class InterestRateSwap:
    def __init__(self,
                 index: str,
                 quote: float,
                 maturity_date: date,
                 purchase_date: date,
                 day_count_convention: str,
                 settlement_convention: str,
                 business_day_calendar: str,
                 business_day_adjustment: str = 'modified following') -> None:
        self.index = index
        self.quote = quote
        self.maturity_date = maturity_date
        self.purchase_date = purchase_date
        self.day_count_convention = day_count_convention
        self._floating_leg: Optional[pd.DataFrame] = None
        self._fixed_leg: Optional[pd.DataFrame] = None
        self.implied_quote = None



    @property
    def floating_leg(self) -> pd.DataFrame:
        return self._floating_leg

    @property
    def fixed_leg(self) -> pd.DataFrame:
        return self._fixed_leg

    def generate_floating_leg_cashflow_schedule(self) -> None:
        columns = ['Leg', 'Payment Number', 'Accrual Start', 'Accrual End', 'Payment Date']
        pass

    def generate_fixed_leg_cashflow_schedule(self) -> None:


        pass
