"""
This script contains the basic risk metric classes.
Unit tests found in [TO BE DETERMINED]
"""

from __future__ import annotations

from datetime import date
from typing import Iterable
from collections.abc import Collection
from dataclasses import dataclass
from fixedIncome.src.curves.curve_enumerations import CurveIndex

ONE_BASIS_POINT = 0.0001


@dataclass
class Risk:
    key_rate_date: date
    pv01: float          # Change in $ per 1 bp bump in underlying CurveIndex at the key_rate_date
    index: CurveIndex


class RiskLadder(Collection[Risk]):
    def __init__(self, risks: Iterable[Risk]) -> None:
        self._risks = list(risks)
        self.iter_index = 0

    @property
    def risks(self) -> list[Risk]:
        return self._risks

    def __contains__(self, risk: Risk) -> bool:
        """ Tests if the Risk ladder contains a specified risk unit."""
        pass

    def __iter__(self):
        return iter(self._risks)

    def __next__(self):
        """ Iterates through the list of Risks. """
        try:
            risk = self.risks[self.iter_index]
        except IndexError:
            self.iter_index = 0
            raise StopIteration()

        self.iter_index += 1
        return risk

    def __len__(self):
        """ Returns the number of Risks"""
        return len(self.risks)

    def __str__(self):
        str_rep = '   date   |   PV01   |   Index   \n'
        str_rep += '-' * 32 + '\n'
        for risk in self:
            risk_str = str(risk.key_rate_date) + '|' + format(risk.pv01, '0.2f') + '|' + risk.index.value + '\n'
            str_rep += risk_str

        return str_rep

    def get_key_rate_dates(self) -> list[date]:
        """ Returns a list of the dates in the Risk Ladder. """
        return [risk.key_rate_date for risk in self]

    def get_pv01s(self) -> list[float]:
        """ Returns a list of the PV01 values in the Risk Ladder. """
        return [risk.pv01 for risk in self]

    def get_indices(self) -> list[CurveIndex]:
        """ Returns a list of the Curve Indices in the Risk Ladder. """
        return [risk.index for risk in self]

