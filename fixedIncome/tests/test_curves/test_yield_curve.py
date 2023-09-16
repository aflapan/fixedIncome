"""
This file contains the unit tests
"""
import datetime
from fixedIncome.src.curves.yield_curve import YieldCurveFactory
from fixedIncome.src.assets.bonds import Bond
from fixedIncome.src.curves.key_rate import KeyRate, KeyRateCollection


# Create a series of test objects.
purchase_date = datetime.date(2023, 2, 27)

# Four Week
four_wk = Bond(price=99.648833,
                coupon=0.00,
                principal=100,
                tenor='1M',
                payment_frequency='zero-coupon',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2023, 3, 28))



one_yr = Bond(price=95.151722,
                coupon=0.00,
                principal=100,
                tenor='1Y',
                payment_frequency='zero-coupon',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2024, 2, 22))

# Two Year
two_yr = Bond(price=99.909356,
                coupon=4.625,
                principal=100,
                tenor='2Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2025, 2, 28))


# Three Year
three_yr = Bond(price=99.795799,
                  coupon=4.0000,
                  principal=100,
                  tenor='3Y',
                  purchase_date=purchase_date,
                  maturity_date=datetime.date(2026, 2, 15))


# Five Year
five_yr = Bond(price=99.511842,
                coupon=4.000,
                principal=100,
                tenor='5Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2028, 2, 28))


# Seven Year
seven_yr = Bond(price=99.625524,
                coupon=4.000,
                principal=100,
                tenor='7Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2030, 2, 28))

# Ten Year
ten_yr = Bond(price=99.058658,
                coupon=3.5000,
                principal=100,
                tenor='10Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2033, 2, 15))

# Twenty Year
twenty_yr = Bond(price=98.601167,
                coupon=3.875,
                principal=100,
                tenor='20Y',
                purchase_date=purchase_date,
                maturity_date=datetime.date(2043, 2, 15))


# Thirty Year
thirty_yr = Bond(price=98.898317,
                   coupon=3.625,
                   principal=100,
                   tenor='30Y',
                   purchase_date=purchase_date,
                   maturity_date=datetime.date(2053, 2, 15))

bond_collection = [four_wk, one_yr, two_yr, three_yr, five_yr, seven_yr, ten_yr, twenty_yr, thirty_yr]


curve_factory_obj = YieldCurveFactory()
yield_curve = curve_factory_obj.construct_yield_curve(bond_collection,
                                                      interpolation_method='cubic',
                                                      reference_date=purchase_date)

#--------------------------------------------------------------------------------------------
# Unit Tests

def test_present_value_for_calibration_instruments() -> None:
    """
    Tests if the present values of the bonds used to calibrate the yield curve
    are equal to the full market price (including accrued interest). The full prices
    were used to calibrate the yield curve, and so we expect equality up to an error threshold.
    """

    pass_thresh = 1e-8

    present_values = [yield_curve.calculate_present_value(bond) for bond in bond_collection]
    full_bond_prices = [bond.full_price for bond in bond_collection]

    assert all([abs(pv - full_price) < pass_thresh for (pv, full_price) in zip(present_values, full_bond_prices)])

def test_bond_pv_does_not_change_for_key_rate_beyond_maturity():
    """
    Tests that a Bond does not have exposure to a KeyRate if the prior date
    for the KeyRate is greater than or equal to the maturity date of the bonds.

    Simply stated, the dv01 of the bonds with respect to a key rate is 0 if the bonds has no
    exposure to the key rate.
    """

    twenty_yr_kr = KeyRate(day_count_convention='act/act',
                           key_rate_date=datetime.date(2043, 2, 15),
                           prior_date=datetime.date(2033, 2, 15),
                           next_date=datetime.date(2053, 2, 15))

    thirty_yr_kr = KeyRate(day_count_convention='act/act',
                           key_rate_date=datetime.date(2053, 2, 15),
                           prior_date=datetime.date(2043, 2, 15),
                           next_date=None)

    pass_thresh = 1e-8
    twenty_yr_key_rate_dv01 = yield_curve.dv01(ten_yr, twenty_yr_kr)
    thirty_yr_key_rate_dv01 = yield_curve.dv01(ten_yr, thirty_yr_kr)

    assert abs(twenty_yr_key_rate_dv01) < pass_thresh and abs(thirty_yr_key_rate_dv01) < pass_thresh


