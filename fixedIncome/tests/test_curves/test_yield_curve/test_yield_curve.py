"""
This file contains the unit tests
"""
import math
from datetime import date
import pandas as pd
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.curves.yield_curves.yield_curve import YieldCurveFactory
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention, PaymentFrequency
from fixedIncome.src.curves.curve_enumerations import InterpolationMethod
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond
from fixedIncome.src.curves.key_rate import KeyRate

# Create a series of test objects.
purchase_date = date(2023, 2, 27)

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

bond_collection = [four_wk, one_yr, two_yr, three_yr, five_yr, seven_yr, ten_yr, twenty_yr, thirty_yr]


curve_factory_obj = YieldCurveFactory()
yield_curve = curve_factory_obj.construct_yield_curve(bond_collection,
                                                      interpolation_method=InterpolationMethod.CUBIC_SPLINE,
                                                      reference_date=purchase_date)

#--------------------------------------------------------------------------------------------
# Unit Tests

def test_present_value_for_calibration_instruments() -> None:
    """
    Tests if the present values of the us_treasury_instruments used to calibrate the yield curve
    are equal to the clean market price (not including accrued interest). The clean prices
    were used to calibrate the yield curve, and so we expect equality up to convergence threshold
    of the calibration.
    """

    pass_thresh = 1e-8
    present_values = [yield_curve.present_value(bond) for bond in bond_collection]
    clean_prices = [bond.price for bond in bond_collection]

    assert all([abs(pv - clean_price) < pass_thresh for (pv, clean_price) in zip(present_values, clean_prices)])

def test_bond_pv_does_not_change_for_key_rate_beyond_maturity() -> None:
    """
    Tests that a Bond does not have exposure to a KeyRate if the prior date
    for the KeyRate is greater than or equal to the maturity date of the us_treasury_instruments.

    Simply stated, the dv01 of the us_treasury_instruments with respect to a key rate is 0 if the
    us_treasury_instruments has no exposure to the key rate.
    """

    twenty_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                           key_rate_date=date(2043, 2, 15),
                           prior_date=date(2033, 2, 15),
                           next_date=date(2053, 2, 15))

    thirty_yr_kr = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
                           key_rate_date=date(2053, 2, 15),
                           prior_date=date(2043, 2, 15),
                           next_date=None)

    pass_thresh = 1e-8
    twenty_yr_key_rate_dv01 = yield_curve.dv01(ten_yr, twenty_yr_kr)
    thirty_yr_key_rate_dv01 = yield_curve.dv01(ten_yr, thirty_yr_kr)

    assert abs(twenty_yr_key_rate_dv01) < pass_thresh and abs(thirty_yr_key_rate_dv01) < pass_thresh

def test_calibrated_yield_curve_is_constant_for_zero_coupon_bonds_fixed_yield() -> None:
    """
    Tests that a curve calibrated on zero-coupon bonds, whose prices are
    generated from a continuously-compounded fixed rate, is a flat curve with value
    equal to the fixed rate.
    """

    FIXED_YIELD = 0.05  # yield is 5%

    one_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                           date(2024, 2, 28),
                                                                                           DayCountConvention.ACTUAL_OVER_ACTUAL))
    one_yr_zc = UsTreasuryBond(price=one_yr_price,
                               coupon_rate=0.00,
                               principal=100,
                               tenor='1Y',
                               payment_frequency=PaymentFrequency.ZERO_COUPON,
                               purchase_date=purchase_date,
                               maturity_date=date(2024, 2, 28))

    two_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                           date(2025, 2, 28),
                                                                                           DayCountConvention.ACTUAL_OVER_ACTUAL))
    two_yr_zc = UsTreasuryBond(price=two_yr_price,
                               coupon_rate=0.00,
                               principal=100,
                               tenor='2Y',
                               payment_frequency=PaymentFrequency.ZERO_COUPON,
                               purchase_date=purchase_date,
                               maturity_date=date(2025, 2, 28))

    three_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                             date(2026, 3, 2),
                                                                                             DayCountConvention.ACTUAL_OVER_ACTUAL))
    three_yr_zc = UsTreasuryBond(price=three_yr_price,
                                 coupon_rate=0.00,
                                 principal=100,
                                 tenor='3Y',
                                 payment_frequency=PaymentFrequency.ZERO_COUPON,
                                 purchase_date=purchase_date,
                                 maturity_date=date(2026, 2, 28))

    four_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                            date(2027, 3, 1),
                                                                                            DayCountConvention.ACTUAL_OVER_ACTUAL))
    four_yr_zc = UsTreasuryBond(price=four_yr_price,
                                coupon_rate=0.00,
                                principal=100,
                                tenor='4Y',
                                payment_frequency=PaymentFrequency.ZERO_COUPON,
                                purchase_date=purchase_date,
                                maturity_date=date(2027, 2, 28))

    five_yr_price = 100 * math.exp(-FIXED_YIELD * DayCountCalculator.compute_accrual_length(purchase_date,
                                                                                            date(2028, 2, 28),
                                                                                            DayCountConvention.ACTUAL_OVER_ACTUAL))
    five_yr_zc = UsTreasuryBond(price=five_yr_price,
                                coupon_rate=0.00,
                                principal=100,
                                tenor='5Y',
                                payment_frequency=PaymentFrequency.ZERO_COUPON,
                                purchase_date=purchase_date,
                                maturity_date=date(2028, 2, 28))

    test_zc_collection = [one_yr_zc, two_yr_zc, three_yr_zc, four_yr_zc, five_yr_zc]
    yield_curve = curve_factory_obj.construct_yield_curve(test_zc_collection,
                                                          interpolation_method=InterpolationMethod.CUBIC_SPLINE,
                                                          reference_date=purchase_date)

    date_range = pd.date_range(start=purchase_date, end=date(2028, 2, 28)).date
    PASS_THRESH = 1E-8

    assert all(abs(yield_curve(date_obj) - FIXED_YIELD) < PASS_THRESH for date_obj in date_range)

