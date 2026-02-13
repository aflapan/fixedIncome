'''
Contains unit tests for the loan object found in
'''
from datetime import date
from dateutil.relativedelta import relativedelta

from fixedIncome.src.assets.credit.loan import Loan, LoanPayment
from fixedIncome.src.scheduling_tools.schedule_enumerations import (PaymentFrequency,
                                                                    PaymentTime,
                                                                    DayCountConvention)



mortgage = Loan(
    principal=745_000 * 0.75,
    interest_rate=0.06125,
    origination_date=date(2025, 3, 11),
    term_length=relativedelta(years=30),
    day_count_convention=DayCountConvention.THIRTY_OVER_THREESIXTY,
    payment_frequency=PaymentFrequency.MONTHLY,
    payment_time=PaymentTime.START_OF_MONTH)

monthly_payment = mortgage.calculate_fixed_payment_amount()
payments = mortgage.generate_payment_schedule(payment_amount=monthly_payment)


def test_number_of_payments_in_loan() -> None:
    '''
    tests that a loan has the correct number of payments.
    '''

    assert 12*30 == len(payments)


def test_sum_of_principal_payments_is_original_principal() -> None:
    '''
    Tests that the sum of principal payments is within a cent of the orignal principal amount.
    '''

    ONE_CENT = 0.01
    sum_of_principal_payments = sum(payment.principal_payment for payment in payments)
    assert abs(mortgage.principal - sum_of_principal_payments) < ONE_CENT
