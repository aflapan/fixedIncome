"""
This module contains the unit tests for
fixedIncome.src.stochastics.short_rate_models.one_factor_models.vasicek_model.py
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.stochastics.short_rate_models.one_factor_models.vasicek_model import VasicekModel


start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
end_time = datetime(2043, 10, 15, 0, 0, 0, 0)

vm = VasicekModel(long_term_mean=0.04,
                  reversion_speed=2.0,
                  volatility=0.02,
                  start_date_time=start_time,
                  end_date_time=end_time)

def test_conditional_mean() -> None:
    """ Tests that the conditional mean is within a tolerance of the long term mean. """
    PASS_THRESH = 1E-4
    assert abs(vm.conditional_mean(0.10, datetime(2050, 12, 25)) - 0.04) < PASS_THRESH


def test_model_evaluates_to_path_on_interpolating_dates() -> None:
    """
    Tests the callable feature of the model correctly gives the path values
    when the datetime object used as an argument is an interpolation date.
    """
    PASS_THRESH = 1E-10
    date_range = pd.date_range(start=start_time, end=end_time, periods=len(vm.path)).to_pydatetime()

    assert all(abs(float(vm(date_time_obj)) - vm.path[0, index]) < PASS_THRESH
               for index, date_time_obj in enumerate(date_range))


def test_affine_yield_coeffs_are_transform_of_price_coeffs() -> None:
    """
    Tests that the affine model coefficients

    Reference: Robonato *Bond Pricing and Yield Curve Modeling: A Structural Approach* pages 165 and 171
    """

    INITIAL_SHORT_RATE = 0.05
    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]
    accruals = [DayCountCalculator.compute_accrual_length(start_time, datetime_obj, DayCountConvention.ACTUAL_OVER_ACTUAL)
                for datetime_obj in admissible_dates]

    PASS_THRESH = 1E-10

    for accrual, date_obj in zip(accruals[1:], admissible_dates[1:]):
        vm._create_bond_price_coeffs(date_obj)
        vm._create_bond_yield_coeffs(date_obj)

        assert abs(vm.price_state_variable_coeffs['intercept']/accrual + vm.yield_state_variable_coeffs['intercept']) < PASS_THRESH
        assert abs(vm.price_state_variable_coeffs['coefficient']/accrual + vm.yield_state_variable_coeffs['coefficient']) < PASS_THRESH