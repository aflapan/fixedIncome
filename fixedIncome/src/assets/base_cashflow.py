"""
A base class to represent a cashflow consisting of payment dates and payment amounts.
"""
from __future__ import annotations
from datetime import date
from typing import Iterable, NamedTuple, Optional
import pandas as pd
from collections.abc import Iterable, MutableSequence, Set
import bisect
from enum import Enum

class Payment(NamedTuple):
    payment_date: date
    payment: Optional[float] = None

class Cashflow(Iterable):
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

    def get_payment_amounts(self) -> list[Optional[float]]:
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
        return pd.Series(self.get_payment_amounts(), index=self.get_payment_dates())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(zip(self.get_payment_dates(), self.get_payment_amounts()),
                            columns=['Payment Dates', 'Payments'])

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

class CashflowKeys(Enum):
    SINGLE_PAYMENT = 'zero coupon'
    FIXED_LEG = 'fixed leg'
    FLOATING_LEG = 'floating leg'
    PROTECTION_LEG = 'protection leg'
    PREMIUM_LEG = 'premium leg'

class CashflowCollection(Set):
    """
    This is the base class from which all financial assets will be subclassed.
    It is a bare-bones template which represents a collection of cashflows.
    """
    def __init__(self, cashflows: Iterable[Cashflow], cashflow_keys: Iterable[CashflowKeys]) -> None:

        self.cashflow_list = list(cashflows)
        self.keys = list(cashflow_keys)
        self.cashflows = {key: cashflow for key, cashflow in zip(self.keys, self.cashflow_list)}

    def __iter__(self):
        """ Iterates through the cashflows in the collection. """
        return self.cashflows.items()


class ZeroCoupon(CashflowCollection):
    def __init__(self, payment_date: date, price: float):
        self._payment_date = payment_date
        self._price = price
        single_payment_iterable = [Payment(self._payment_date, 1.0)]
        cashflows = [Cashflow(single_payment_iterable)]  # a singleton cashflow of $1
        cashflow_keys = [CashflowKeys.SINGLE_PAYMENT]
        super().__init__(cashflows, cashflow_keys)

    @property
    def price(self):
        return self._price
    @property
    def payment_date(self):
        return self._payment_date

    def __len__(self):
        return len(self.cashflows)

    def __contains__(self, key: CashflowKeys) -> bool:
        """
        Returns whether the cash flow collection contains the provided Key
        in the dictionary of key: Cashflow pairings.
        """
        return key in self.cashflows

    def to_knot_value_pair(self) -> tuple[date, float]:
        """
        Converts the zero-coupon bond to a (date, float)
        tuple to be used by curves to interpolate.
        """
        return self.payment_date, self.price




