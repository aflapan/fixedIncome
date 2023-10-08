"""
This file contains unit tests for the base curve objects found in
fixedIncome.src.curves.base_curve.py
"""
import pytest
from datetime import date
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.assets.base_cashflow import ZeroCoupon
from fixedIncome.src.curves.base_curve import (KnotValuePair,
                                               Curve,
                                               EndBehavior,
                                               InterpolationMethod,
                                               CurveIndex,
                                               DiscountCurve)

from fixedIncome.src.curves.key_rate import KeyRate


PASS_THRESH = 1E-8

knot_points = [
    KnotValuePair(date(2023, 9, 18), 1.0),
    KnotValuePair(date(2023, 9, 19), 2.0),
    KnotValuePair(date(2023, 9, 20), 1.0),
    KnotValuePair(date(2023, 9, 21), 0.0)
]

curve_obj = Curve(interpolation_values=knot_points,
                  interpolation_method=InterpolationMethod.LINEAR,
                  interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
                  reference_date=None,
                  left_end_behavior=EndBehavior.CONSTANT,
                  right_end_behavior=EndBehavior.CONSTANT)


error_curve = Curve(interpolation_values=knot_points,
                    interpolation_method=InterpolationMethod.LINEAR,
                    interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
                    reference_date=None,
                    left_end_behavior=EndBehavior.ERROR,
                    right_end_behavior=EndBehavior.ERROR)

key_rate = KeyRate(day_count_convention=DayCountConvention.ACTUAL_OVER_360,
                   key_rate_date=date(2023, 9, 20),
                   prior_date=date(2023, 9, 19),
                   next_date=date(2023, 9, 21))


def test_curve_object_instantiates() -> None:
    """ Tests that the curve object can be constructed using valid inputs. """

    assert isinstance(curve_obj, Curve)


def test_curve_object_is_callable() -> None:
    """ Tests that the curve object is callable and returns the correct value. """
    assert abs(curve_obj(date(2023, 9, 20)) - 1.0) < PASS_THRESH


def test_constant_extrapolation_for_left_endpoint() -> None:
    """
    Tests that a value less than the first knot data evaluates to the first knot value
    when the left_end_behavior is set to CONSTANT.
    """
    assert abs(curve_obj(date(2020, 1, 1)) - 1.0) < PASS_THRESH


def test_constant_extrapolation_for_right_endpoint() -> None:
    """
    Tests that a value less than the first knot data evaluates to the first knot value
    when the right_end_behavior is set to CONSTANT.
    """
    assert abs(curve_obj(date(2030, 1, 1)) - 0.0) < PASS_THRESH


def test_error_extrapolation_for_left_endpoint() -> None:
    """
    Tests that a ValueError is raised
    """
    first_date = knot_points[0].knot
    input_date = date(2020, 1, 1)

    with pytest.raises(ValueError):
        error_curve(input_date)


def test_error_extrapolation_for_right_endpoint() -> None:
    """
    Tests that a ValueError is raised when a date exceeds the last interpolation
    knot date of a curve and the right_end_behavior parameter is set to ERROR.
    """
    last_date = knot_points[-1].knot
    input_date = date(2030, 1, 1)

    with pytest.raises(ValueError):
        error_curve(input_date)

def test_curve_can_evaluate_with_key_rate_adjustment() -> None:
    """
    Tests that one can call the curve object with both a date and
    an adjustment function and obtain the desired output.
    """
    target_val = 1.0 + key_rate.bump_val
    assert abs(curve_obj(key_rate.key_rate_date, adjustment=key_rate) - target_val) < PASS_THRESH


#-------------------------------------------------------------------
# Test discount curve
REFERENCE_DATE = date(2023, 9, 1)

first_zc = ZeroCoupon(payment_date=date(2023, 12, 25), price=0.99)
second_zc = ZeroCoupon(payment_date=date(2024, 1, 30), price=0.98)
third_zc = ZeroCoupon(payment_date=date(2027, 1, 1), price=0.80)
fourth_zc = ZeroCoupon(payment_date=date(2030, 1, 1), price=0.50)

zero_coupon_bonds = [first_zc, second_zc, third_zc, fourth_zc]
interpolation_values = [zc.to_knot_value_pair() for zc in zero_coupon_bonds]

discount_curve = DiscountCurve(interpolation_values=interpolation_values,
                               interpolation_method=InterpolationMethod.LINEAR,
                               index=CurveIndex.NONE,
                               interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_360,
                               reference_date=REFERENCE_DATE)

def test_calling_discount_curve_prices_zero_coupons_exactly() -> None:
    """
    Tests that the discount curve reprices the interpolated
    zero-coupon prices within numerical error.
    """
    assert all(abs(discount_curve(zc.payment_date) - zc.price) < PASS_THRESH
               for zc in zero_coupon_bonds)


def test_present_value_of_zero_coupon_cashflows_are_prices() -> None:
    """
    Tests that the present value of the cashflow in each zero coupon bond
    equals its price.
    """
    assert all(abs(discount_curve.present_value(zc) - zc.price) < PASS_THRESH
           for zc in zero_coupon_bonds)


def test_discount_curve_reprices_zero_coupons_for_all_interpolation_methods_and_day_count_conventions() -> None:
    """
    Tests that the zero coupon prices are recreated within numerical precision
    for each discount interpolation method and day count convention.
    """
    methods_repriced_exactly = []
    for method in InterpolationMethod:
        for dcc in DayCountConvention:
            curve = DiscountCurve(interpolation_values=interpolation_values,
                                  interpolation_method=method,
                                  index=CurveIndex.NONE,
                                  interpolation_day_count_convention=dcc,
                                  reference_date=REFERENCE_DATE)

            method_passed = all(abs(curve.present_value(zc) - zc.price) < PASS_THRESH
                                for zc in zero_coupon_bonds)
            methods_repriced_exactly.append(method_passed)

    assert all(methods_repriced_exactly)

def test_length_of_zero_coupon_is_one() -> None:
    """ Tests that the __len__ method correctly returns 1 for a
    zero-coupon bond.
    """
    assert len(first_zc) == 1


