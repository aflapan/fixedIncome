from datetime import date, datetime, timedelta
import calendar
from dateutil.relativedelta import relativedelta
from typing import Optional, Iterable
import itertools
from dataclasses import dataclass
import matplotlib.pyplot as plt

from fixedIncome.src.assets.base_cashflow import CashflowCollection, CashflowKeys, Cashflow, Payment
from fixedIncome.src.curves.base_curve import DiscountCurve, KnotValuePair
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.scheduling_tools.schedule_enumerations import (PaymentFrequency,
                                                                    PaymentTime,
                                                                    BusinessDayAdjustment,
                                                                    DayCountConvention,
                                                                    SettlementConvention,
                                                                    ImmMonths,
                                                                    Months)

@dataclass
class LoanPayment(Payment):
    principal_payment: float
    interest_payment: float = float('nan')


class Loan(CashflowCollection):

    def __init__(self,
                 principal: float,
                 interest_rate: float,
                 origination_date: date,
                 term_length: timedelta | relativedelta,
                 day_count_convention: DayCountConvention = DayCountConvention.THIRTY_OVER_THREESIXTY,
                 payment_frequency: PaymentFrequency = PaymentFrequency.MONTHLY,
                 payment_time: PaymentTime = PaymentTime.END_OF_MONTH) -> None:

        self._principal = principal
        self._interest_rate = interest_rate
        self._origination_date = origination_date
        self._term_length = term_length
        self._day_count_convention = day_count_convention
        self._payment_frequency = payment_frequency
        self._payment_time = payment_time

        self._end_date = self._origination_date + self._term_length

    @property
    def principal(self) -> float | int:
        ''' the principal amount of the loan.'''
        return self._principal

    def calculate_fixed_payment_amount(self) -> float:
        '''
        The solution comes from the equation

        Payment sum_{j=0}^{N}
        '''
        length_in_years = DayCountCalculator.compute_accrual_length(
            self._origination_date, self._end_date, dcc=self._day_count_convention
        )

        compound_factor = (1+self._interest_rate/self._payment_frequency.value)**(length_in_years * self._payment_frequency.value)
        numerator = self.principal * compound_factor
        denominator = (1 - compound_factor)/(-self._interest_rate / self._payment_frequency.value)

        return numerator/denominator



    def generate_payment_schedule(self, payment_amount: int | float, extra_payments: Optional[float|Iterable] = None) -> list[date]:
        """ Generates the schedule of payments. """
        payment_dates = []
        payments = []
        start_date = self._origination_date
        remaining_principal = self.principal

        # adjust start date
        match self._payment_time:

            case PaymentTime.END_OF_MONTH:
                _, last_day = calendar.monthrange(start_date.year, 11)
                start_date = date(start_date.year, start_date.month, last_day)

            case PaymentTime.START_OF_MONTH:
                if start_date.month == 12:
                    start_date = date(start_date.year + 1, 1, 1)
                else:
                    start_date = date(start_date.year, start_date.month + 1, 1)
            case _:
                raise ValueError(f'{self._payment_time} is not a valid payment time for a loan.')

        end_date = start_date + self._term_length

        match self._payment_frequency:
            case PaymentFrequency.MONTHLY:
                payment_increment = relativedelta(months=1)
            case PaymentFrequency.QUARTERLY:
                payment_increment = relativedelta(months=3)
            case PaymentFrequency.SEMI_ANNUAL:
                payment_increment = relativedelta(months=6)
            case PaymentFrequency.ANNUAL:
                payment_increment = relativedelta(years=1)
            case _:
                raise ValueError(f'{self._payment_frequency} is not a valid payment frequency for a loan.')

        payment_date = start_date
        while payment_date <= end_date:
            payment_dates.append(payment_date)
            payment_date += payment_increment

        # unpack extra payments
        if extra_payments is not None:
            if isinstance(extra_payments, float):
                extra_payments = [extra_payments for payment in payment_dates]
            else:
                extra_payments = list(extra_payments)
                if len(extra_payments) <= len(payment_dates):
                    extra_payments = extra_payments + [0.0 for payment in range(len(extra_payments), len(payment_dates))]
                else:
                    extra_payments = extra_payments[slice(0, len(payment_dates))]
        else:
            extra_payments = [0.0 for payment_date in payment_dates]

        for extra_payment, (sd, ed) in zip(extra_payments, itertools.pairwise(payment_dates)):

            interest_payment = self._calculate_interest_payment(
                accrual_start=sd,
                accrual_end=ed,
                interest_rate=self._interest_rate,
                principal=remaining_principal
            )

            principal_payment_amount = payment_amount - interest_payment + extra_payment

            if remaining_principal <= principal_payment_amount:
                principal_payment_amount = remaining_principal
                payment_amount = principal_payment_amount + interest_payment

            payments.append(
                LoanPayment(payment_date=ed, payment=payment_amount,
                            principal_payment=principal_payment_amount,
                            interest_payment=interest_payment
                            )
            )
            remaining_principal -= principal_payment_amount

        return payments


    def _calculate_interest_payment(self,
                                    accrual_start: date,
                                    accrual_end: date,
                                    interest_rate: Optional[float] = None,
                                    principal: Optional[int | float] = None) -> float:
        ''' Calculates the interest payment on the principal amount for a specified accrual start and end date. '''
        if interest_rate is None:
            interest_rate = self._interest_rate

        if principal is None:
            principal = self.principal

        accrual_factor = DayCountCalculator.compute_accrual_length(
            accrual_start, accrual_end, dcc=self._day_count_convention
        )

        return accrual_factor * interest_rate * principal


    def plot_amortization_schedule(self, payment_amount: int | float,  extra_payments: Optional[float|Iterable] = None) -> None:
        ''' Plots the amortization schedule'''
        payment_schedule = self.generate_payment_schedule(payment_amount=payment_amount, extra_payments=extra_payments)
        plt.figure(figsize=(15, 5))
        plt.plot([payment.payment_date for payment in payment_schedule],
                 [payment.payment for payment in payment_schedule])

        plt.plot([payment.payment_date for payment in payment_schedule],
                 [payment.principal_payment for payment in payment_schedule])

        plt.plot([payment.payment_date for payment in payment_schedule],
                 [payment.interest_payment for payment in payment_schedule])
        plt.legend(['Payment', 'Principal', 'Interest'], frameon=False)
        plt.xlabel('Date')
        plt.ylabel('USD ($)')
        plt.grid(alpha=0.25)

        plt.show()


    def present_value(self, discount_curve: DiscountCurve) -> float:
        ''' Calculates the present value of the loan payments.'''
        payments = self.generate_payment_schedule()
        return sum(discount_curve.present_value(payment) for payment in payments)

    def to_knot_value_pair(self):
        pass



if __name__ == '__main__':

    mortgage = Loan(
        principal=745_000 * 0.75,
        interest_rate=0.06125,
        origination_date=date(2025, 3, 11),
        term_length=relativedelta(years=30),
        day_count_convention=DayCountConvention.THIRTY_OVER_THREESIXTY,
        payment_frequency=PaymentFrequency.MONTHLY,
        payment_time=PaymentTime.START_OF_MONTH)

    monthly_payment = mortgage.calculate_fixed_payment_amount()

    print(monthly_payment)
    payments = mortgage.generate_payment_schedule(payment_amount=monthly_payment, extra_payments=None)

    print('\n'.join([str(payment) for payment in payments]))

    print('Original principal is: ', mortgage.principal)
    print('Sum of principal payments is: ', sum(payment.principal_payment for payment in payments))

    mortgage.plot_amortization_schedule(payment_amount=monthly_payment, extra_payments=None)

    print('Total interest paid: ', sum(payment.interest_payment for payment in payments))


