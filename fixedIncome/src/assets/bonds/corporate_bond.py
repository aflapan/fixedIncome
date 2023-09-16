from fixedIncome.src.assets.bonds.bond import Bond
from typing import Optional
from datetime import date


class CorporateBond(Bond):
    """
    Class for corporate bonds which subclasses from the base Bond class.
    """
    def __init__(self, yield_spread: Optional[float], issuer: str, credit_rating: str,  **kwargs):
        super().__init__(**kwargs)
        self.yield_spread = yield_spread
        self.issuer = issuer
        self.credit_rating = credit_rating

    def calculate_yield_to_maturity(self, purchase_date: Optional[date] = None) -> float:
        """
        Calculates the yield to maturity
        """
        ytm = super().calculate_yield_to_maturity(purchase_date=purchase_date)
        return ytm - self.yield_spread if self.yield_spread is not None else ytm
