"""
This file contains unit tests for the UsTreasuryBond class found in

fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments
"""

from datetime import date
from fixedIncome.src.scheduling_tools.schedule_enumerations import PaymentFrequency
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond

#-----------------------------------------------------------------
# Create a series of test objects.
purchase_date = date(2023, 2, 27)

# Four Week
four_wk = UsTreasuryBond(price=99.648833,
                         coupon=0.00,
                         principal=100,
                         tenor='1M',
                         payment_frequency=PaymentFrequency.ZERO_COUPON,
                         purchase_date=purchase_date,
                         maturity_date=date(2023, 3, 28))


one_yr = UsTreasuryBond(price=95.151722,
                        coupon=0.00,
                        principal=100,
                        tenor='1Y',
                        payment_frequency=PaymentFrequency.ZERO_COUPON,
                        purchase_date=purchase_date,
                        maturity_date=date(2024, 2, 22))

# Two Year
two_yr = UsTreasuryBond(price=99.909356,
                        coupon=4.625,
                        principal=100,
                        tenor='2Y',
                        purchase_date=purchase_date,
                        maturity_date=date(2025, 2, 28))


# Three Year
three_yr = UsTreasuryBond(price=99.795799,
                          coupon=4.0000,
                          principal=100,
                          tenor='3Y',
                          purchase_date=purchase_date,
                          maturity_date=date(2026, 2, 15))


# Five Year
five_yr = UsTreasuryBond(price=99.511842,
                         coupon=4.000,
                         principal=100,
                         tenor='5Y',
                         purchase_date=purchase_date,
                         maturity_date=date(2028, 2, 28))


# Seven Year
seven_yr = UsTreasuryBond(price=99.625524,
                          coupon=4.000,
                          principal=100,
                          tenor='7Y',
                          purchase_date=purchase_date,
                          maturity_date=date(2030, 2, 28))

# Ten Year
ten_yr = UsTreasuryBond(price=99.058658,
                        coupon=3.5000,
                        principal=100,
                        tenor='10Y',
                        purchase_date=purchase_date,
                        maturity_date=date(2033, 2, 15))

# Twenty Year
twenty_yr = UsTreasuryBond(price=98.601167,
                           coupon=3.875,
                           principal=100,
                           tenor='20Y',
                           purchase_date=purchase_date,
                           maturity_date=date(2043, 2, 15))


# Thirty Year
thirty_yr = UsTreasuryBond(price=98.898317,
                           coupon=3.625,
                           principal=100,
                           tenor='30Y',
                           purchase_date=purchase_date,
                           maturity_date=date(2053, 2, 15))

# all us_treasury_instruments with semi-annual coupons
long_term_bond_collection = [two_yr, three_yr, five_yr, seven_yr, ten_yr, twenty_yr, thirty_yr]

#------------------------------------------------------------------------
# Unit Tests

def test_accrued_interest_example():
    """
    This test computes the accrued interest on an example Treasury Bond and compares the result to the known answer.
    The example is taken from Tuckman and Serrat, *Fixed Income Securities, 4th ed.*, pages 60-61.
    """

    book_answer = 15.711
    PASS_THRESH = 0.005  # less than a half of a cent off, within rounding error

    test_bond = UsTreasuryBond(price=10_000,  # testing accrued interest, price does not matter here.
                               coupon=0.625,
                               principal=10_000,
                               tenor='10Y',
                               purchase_date=date(2021, 5, 14),  # settlement day corresponds to May 17th, 2021
                               maturity_date=date(2030, 8, 15))

    test_bond.calculate_accrued_interest()

    assert abs(test_bond.accrued_interest - book_answer) < PASS_THRESH


def test_accrued_interest_nonnegative():
    """
    Checks that all accrued interest values are non-negative.
    """

    accrued_interests = [bond.accrued_interest for bond in long_term_bond_collection]

    assert all([accrued_interest >= 0.0 for accrued_interest in accrued_interests])


def test_yield_to_maturity_present_value():
    """
    Tests if the present values of us_treasury_instruments cashflows equal the full market price when
     semi-annualy discounted at their yield to maturity rates.

     Should assert True, as the yield to maturity is defined as
     the single rate at which discounting cashflow payments results in the full market price.
    """

    PASS_THRESH = 1e-10

    ytms = [bond.calculate_yield_to_maturity() for bond in long_term_bond_collection]

    present_values = [bond.calculate_present_value_for_fixed_yield(ytm) for (bond, ytm)
                      in zip(long_term_bond_collection, ytms)]

    full_prices = [bond.get_full_price() for bond in long_term_bond_collection]

    assert all([abs(pv - fp) < PASS_THRESH for (pv, fp) in zip(present_values, full_prices)])

