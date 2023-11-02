"""
This file contains unit tests for the UsTreasuryBond class found in

fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments
"""

from datetime import date
from fixedIncome.src.scheduling_tools.schedule_enumerations import PaymentFrequency
from fixedIncome.src.assets.base_cashflow import CashflowKeys
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond

#-----------------------------------------------------------------
# Create a series of test objects.
purchase_date = date(2023, 2, 27)

# Two Year
# Four Week
four_wk = UsTreasuryBond(price=99.648833,
                         coupon_rate=0.00,
                         principal=100,
                         tenor='1M',
                         payment_frequency=PaymentFrequency.ZERO_COUPON,
                         purchase_date=purchase_date,
                         maturity_date=date(2023, 3, 28))


one_yr = UsTreasuryBond(price=95.151722,
                        coupon_rate=0.00,
                        principal=100,
                        tenor='1Y',
                        payment_frequency=PaymentFrequency.ZERO_COUPON,
                        purchase_date=purchase_date,
                        maturity_date=date(2024, 2, 22))

# Two Year
two_yr = UsTreasuryBond(price=99.909356,
                        coupon_rate=4.625,
                        principal=100,
                        tenor='2Y',
                        purchase_date=purchase_date,
                        maturity_date=date(2025, 2, 28))


# Three Year
three_yr = UsTreasuryBond(price=99.795799,
                          coupon_rate=4.0000,
                          principal=100,
                          tenor='3Y',
                          purchase_date=purchase_date,
                          maturity_date=date(2026, 2, 15))


# Five Year
five_yr = UsTreasuryBond(price=99.511842,
                         coupon_rate=4.000,
                         principal=100,
                         tenor='5Y',
                         purchase_date=purchase_date,
                         maturity_date=date(2028, 2, 28))


# Seven Year
seven_yr = UsTreasuryBond(price=99.625524,
                          coupon_rate=4.000,
                          principal=100,
                          tenor='7Y',
                          purchase_date=purchase_date,
                          maturity_date=date(2030, 2, 28))

# Ten Year
ten_yr = UsTreasuryBond(price=99.058658,
                        coupon_rate=3.5000,
                        principal=100,
                        tenor='10Y',
                        purchase_date=purchase_date,
                        maturity_date=date(2033, 2, 15))

# Twenty Year
twenty_yr = UsTreasuryBond(price=98.601167,
                           coupon_rate=3.875,
                           principal=100,
                           tenor='20Y',
                           purchase_date=purchase_date,
                           maturity_date=date(2043, 2, 15))


# Thirty Year
thirty_yr = UsTreasuryBond(price=98.898317,
                           coupon_rate=3.625,
                           principal=100,
                           tenor='30Y',
                           purchase_date=purchase_date,
                           maturity_date=date(2053, 2, 15))

# all us_treasury_instruments with semi-annual coupons
long_term_bond_collection = [two_yr, three_yr, five_yr, seven_yr, ten_yr, twenty_yr, thirty_yr]

#------------------------------------------------------------------------
# Unit Tests

def test_accrued_interest_example() -> None:
    """
    This test computes the accrued interest on an example Treasury Bond and compares the result to the known answer.
    The example is taken from Tuckman and Serrat, *Fixed Income Securities, 4th ed.*, pages 60-61.
    """

    book_answer = 15.711
    PASS_THRESH = 0.005  # less than a half of a cent off, within rounding error

    test_bond = UsTreasuryBond(price=10_000,  # testing accrued interest, price does not matter here.
                               coupon_rate=0.625,
                               principal=10_000,
                               tenor='10Y',
                               purchase_date=date(2021, 5, 14),  # settlement day corresponds to May 17th, 2021
                               maturity_date=date(2030, 8, 15))

    test_bond.calculate_accrued_interest()

    assert abs(test_bond.accrued_interest - book_answer) < PASS_THRESH


def test_accrued_interest_is_nonnegative() -> None:
    """
    Checks that all accrued interest values are non-negative.
    """

    accrued_interests = [bond.accrued_interest for bond in long_term_bond_collection]

    assert all([accrued_interest >= 0.0 for accrued_interest in accrued_interests])


def test_yield_to_maturity_present_value() -> None:
    """
    Tests if the present values of us_treasury_instruments cashflows equal the full market price when
     semi-annualy discounted at their yield to maturity rates.

     Should assert True, as the yield to maturity is defined as
     the single rate at which discounting cashflow payments results in the full market price.
    """

    PASS_THRESH = 1e-10

    ytms = (bond.yield_to_maturity() for bond in long_term_bond_collection)

    present_values = (bond.discount_cashflows_by_fixed_rate(ytm) for (bond, ytm)
                      in zip(long_term_bond_collection, ytms))

    full_prices = (bond.get_full_price() for bond in long_term_bond_collection)

    assert all(abs(pv - fp) < PASS_THRESH for (pv, fp) in zip(present_values, full_prices))


def test_that_cashflow_coupon_payments_align_with_schedule() -> None:
    """
    Tests that the number of coupon payments in each bond's COUPON_PAYMENTS cashflows
    is equal to the expected number (two coupons per year for the duration of the bond.)
    """
    num_coupon_payments = (4, 6, 10, 14, 20, 40, 60)

    assert all(len(bond[CashflowKeys.COUPON_PAYMENTS]) == num_coupons
               for bond, num_coupons in zip(long_term_bond_collection, num_coupon_payments))


def test_principal_repayment_date_equals_last_coupon_date() -> None:
    """
    Tests that the principal is repaid on the last coupon payment for a
    coupon-paying bond.
    """
    assert all(bond[CashflowKeys.COUPON_PAYMENTS][-1].payment_date == bond[CashflowKeys.SINGLE_PAYMENT][0].payment_date
               for bond in long_term_bond_collection)

def test_discount_cashflows_by_zero_returns_sum_of_coupon_and_principal() -> None:
    """
    Tests that when discounting a bond using semi-annual compounding at a rate of 0%
    results in a present value which is just the sum of all payments.
    """
    PASS_THRESH = 1E-10
    test_value = thirty_yr.discount_cashflows_by_fixed_rate(fixed_rate=0.0)
    coupon_sum = sum(payment.payment for payment in thirty_yr[CashflowKeys.COUPON_PAYMENTS])
    principal_sum = sum(payment.payment for payment in thirty_yr[CashflowKeys.SINGLE_PAYMENT])
    total_cashflow_sum = coupon_sum + principal_sum
    assert abs(total_cashflow_sum - test_value) < PASS_THRESH

def test_yield_to_maturity_computes() -> None:
    """
    """
    ten_yr.yield_to_maturity()
    assert True
