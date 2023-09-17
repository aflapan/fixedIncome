"""
A base class to represent a cashflow consisting of payment dates and payment amounts.
"""
from __future__ import annotations
from datetime import date
from typing import Iterable, NamedTuple, Optional
import pandas as pd
from collections.abc import Iterable, MutableSequence
import bisect

class Payment(NamedTuple):
    payment_date: date
    payment: Optional[float] = None


class Cashflow(Iterable, MutableSequence):
    """ A base class representing a cashflows. """
    def __init__(self, payments: Iterable[Payment]) -> None:
        self._schedule = sorted(list(payments), key=lambda payment: payment.payment_date)

    @property
    def schedule(self) -> list[Payment]:
        return self._schedule

    def __iter__(self):
        return iter(self.schedule)

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, item: int) -> Payment:
        return self.schedule[item]

    def get_payments(self) -> list[Optional[float]]:
        """ Returns a list of the payment amounts """
        return [payment.payment for payment in self.schedule]

    def get_payment_dates(self) -> list[date]:
        """ Returns a list of the payment dates. """
        return [payment.payment_date for payment in self.schedule]

    def add_payment(self, payment: Payment) -> None:
        """ Adds a payment to the payoff schedule. Modifies the object in place.
        Assumes the payments are sorted by payment date.
        """
        bisect.insort_right(self._schedule, payment, key=lambda payoff: payoff.payment_date)

    # Conversion methodsan introduction to git
    def to_series(self) -> pd.Series:
        return pd.Series(self.get_payments(), index=self.get_payment_dates())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(zip(self.get_payment_dates(), self.get_payments()), columns=['Payment Dates', 'Payments'])

    @classmethod
    def create_from_date_and_float_iterables(cls,
                                             payment_dates: Iterable[date],
                                             payments: Iterable[Optional[float]]) -> Cashflow:
        """ Instantiates a Cashflow from separate iterables of payment dates and amounts. """
        payment_dates = list(payment_dates)
        payments = list(payments)

        if len(payment_dates) != len(payments):
            raise ValueError()

        return cls(payments=(Payment(payment_date, payment) for payment_date, payment in zip(payment_dates, payments)))


