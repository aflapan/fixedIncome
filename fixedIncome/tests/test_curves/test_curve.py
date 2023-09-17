"""
This file contains unit tests for the base curve object.
"""
import pytest
from datetime import date
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention, DayCountCalculator
from fixedIncome.src.curves.curve import Curve, KnotValuePair, InterpolationMethod, EndBehavior

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
    Tests that a ValueError is raised
    """
    last_date = knot_points[-1].knot
    input_date = date(2030, 1, 1)

    with pytest.raises(ValueError):
        error_curve(input_date)


