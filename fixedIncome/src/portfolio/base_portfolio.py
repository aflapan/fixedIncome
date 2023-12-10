
from typing import Iterable, NamedTuple
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.curves.base_curve import DiscountCurve
from fixedIncome.src.assets.base_cashflow import CashflowCollection
from fixedIncome.src.risk.key_rate import KeyRate, KeyRateCollection

class PortfolioEntry(NamedTuple):
    quantity: float
    asset: CashflowCollection


class Portfolio:
    """
    A base class representing a portfolio of assets, i.e. CashflowCollection objects.
    """
    def __init__(self, assets: Iterable[PortfolioEntry]) -> None:
        self._assets = list(assets)
        self._iter_index = 0

    @property
    def assets(self) -> list[PortfolioEntry]:
        return self._assets

    def __iter__(self):
        """ """
        return (asset for asset in self.assets)
    def __next__(self) -> PortfolioEntry:
        """ """
        try:
            next_asset = self.assets[self._iter_index]
        except IndexError:
            self._iter_index = 0
            raise StopIteration()
        self._iter_index += 1
        return next_asset


    def present_value(self, curve: DiscountCurve) -> float:
        """
        Calculates the present value of the portfolio on a given discount curve by
        summing the present values of the individual positions times the quantity held.
        """
        return sum(entry.quantity * entry.asset.present_value(curve) for entry in self.assets)

    def to_key_rate_collection(self, day_count_convention: DayCountConvention) -> KeyRateCollection:
        """
        Takes a portfolio and returns a KeyRateCollection, where each
        asset in the portfolio has a corresponding KeyRate in the collection.
        """
        kr_list = [KeyRate(day_count_convention=day_count_convention,
                           key_rate_date=portfolio_asset.asset.to_knot_value_pair().knot) for portfolio_asset in self]

        return KeyRateCollection(kr_list)

