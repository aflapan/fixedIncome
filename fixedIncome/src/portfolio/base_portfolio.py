
from typing import Iterable, NamedTuple
from fixedIncome.src.curves.base_curve import DiscountCurve
from fixedIncome.src.assets.base_cashflow import CashflowCollection


class PortfolioEntry(NamedTuple):
    quantity: float
    asset: CashflowCollection


class Portfolio:
    """
    A base class representing a portfolio of assets, i.e. CashflowCollection objects.
    """
    def __init__(self, assets: Iterable[PortfolioEntry]) -> None:
        self._assets = list(assets)

    @property
    def assets(self) -> list[PortfolioEntry]:
        return self._assets

    def present_value(self, curve: DiscountCurve) -> float:
        """ Calculates the present value of the portfolio on a given discount curve by
        summing the present values of the individual positions times the quantity held.
        """
        return sum(entry.quantity * entry.asset.present_value(curve) for entry in self.assets)









