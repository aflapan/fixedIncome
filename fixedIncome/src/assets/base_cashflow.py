"""
A base class to represent a cashflow consisting of payment dates and payment amounts.

Unit tests contained in the script
"""
from __future__ import annotations
from datetime import date
import operator
from typing import Optional
from dataclasses import dataclass
import pandas as pd
from collections.abc import Iterable, Set
import bisect
from enum import Enum
from abc import abstractmethod
from fixedIncome.src.curves.base_curve import DiscountCurve, KnotValuePair

@dataclass
class Payment:
    payment_date: date
    payment: Optional[float]


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

    def __getitem__(self, key: int) -> Payment | Cashflow:
        """  Retrieves a Cashflow or Payment object from the collection via indexing. """
        if isinstance(key, slice):
            cls = type(self)
            return cls(self.schedule[key])

        index = operator.index(key)
        return self.schedule[index]

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

    # Conversion methods
    def to_series(self) -> pd.Series:
        return pd.Series(self.get_payment_amounts(), index=self.get_payment_dates())

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(zip(self.get_payment_dates(), self.get_payment_amounts()),
                            columns=['Payment Dates', 'Payments'])

    def present_value(self, discount_curve: DiscountCurve) -> float:
        """
        Calculates the present value of the cash flow based on discounting
        payment amounts using the provided discount curve.
        """
        return sum(discount_curve(payment.payment_date) * payment.payment
                   for payment in self if payment.payment is not None
                   and payment.payment_date >= discount_curve.reference_date)

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
    SINGLE_PAYMENT = 0
    COUPON_PAYMENTS = 1
    FIXED_LEG = 2
    FLOATING_LEG = 3
    PROTECTION_LEG = 4
    PREMIUM_LEG = 5
    MARK_TO_MARKET = 6

class CashflowCollection(Set):
    """
    This is the base class from which all financial assets will be subclassed.
    It is a bare-bones template which represents a collection of cashflows.
    """
    def __init__(self, cashflows: Iterable[Optional[Cashflow]], cashflow_keys: Iterable[CashflowKeys]) -> None:

        self.cashflow_list = list(cashflows)
        self.keys = list(cashflow_keys)
        self.cashflows = {key: cashflow for key, cashflow in zip(self.keys, self.cashflow_list)}

    def __len__(self):
        """ Returns the number of cashflows in the collection. """
        return len(self.cashflows)

    def __contains__(self, key: CashflowKeys) -> bool:
        """
        Returns whether the cash flow collection contains the provided Key
        in the dictionary of key: Cashflow pairings.
        """
        return key in self.cashflows
    def __iter__(self):
        """ Iterates through the cashflows in the collection. """
        return self.cashflows.items()

    def __getitem__(self, item: CashflowKeys | str) -> Cashflow:
        """
        Allows one to index by the Cashflow key
        and obtain the corresponding individual cash flow.
        """
        try:
            return self.cashflows[item]
        except KeyError:
            return self.cashflows[item.value]

    def __setitem__(self, cashflow_key: CashflowKeys, new_cashflow: Cashflow) -> None:
        """
        Method to put a new cashflow into the Cashflow collection based on key.
        """
        if cashflow_key in self.cashflows:
            self.cashflows[cashflow_key] = new_cashflow
        else:
            self.cashflow_list.append(new_cashflow)
            self.keys.append(cashflow_key)
            self.cashflows[cashflow_key] = new_cashflow


    @abstractmethod
    def to_knot_value_pair(self) -> KnotValuePair:
        """
        An abstract method which unambiguously maps the cash flow collection
        to a knot-value pair to be used in curve interpolation.
        """
        return NotImplemented('Please provide an implementation for to_knot_value_pair.')

    @abstractmethod
    def present_value(self, discount_curve: DiscountCurve) -> float:
        """
        An abstract method which provides an instrument-specific
        method for computing present values of the cash flow collection
        on a discount curve.
        """
        return NotImplemented('Please provide an implementation for present_value.')


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


    def to_knot_value_pair(self) -> KnotValuePair:
        """
        Converts the zero-coupon bond to a KnotValuePair
        NamedTuple to be used by curves to interpolate.
        """
        return KnotValuePair(knot=self.payment_date, value=self.price)

    def present_value(self, discount_curve: DiscountCurve) -> float:
        """
        Calculates the present value of the zero-coupon payment amount
        on the provided discount curve.
        """
        return self[CashflowKeys.SINGLE_PAYMENT].present_value(discount_curve)

