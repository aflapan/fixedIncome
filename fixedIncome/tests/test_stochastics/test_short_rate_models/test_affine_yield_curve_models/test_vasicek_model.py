"""
This module contains the unit tests for
fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model.py
"""
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import pytest

from fixedIncome.src.scheduling_tools.scheduler import Scheduler
from fixedIncome.src.scheduling_tools.schedule_enumerations import DayCountConvention
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator
from fixedIncome.src.stochastics.brownian_motion import BrownianMotion
from fixedIncome.src.stochastics.short_rate_models.affine_yield_curve_models.vasicek_model import VasicekModel, MultivariateVasicekModel


start_time = datetime(2023, 10, 15, 0, 0, 0, 0)
end_time = datetime(2053, 10, 15, 0, 0, 0, 0)

brownian_motion = BrownianMotion(start_date_time=start_time,
                                 end_date_time=end_time,
                                 dimension=1)

vm = VasicekModel(reversion_level=0.04,
                  reversion_speed=2.0,
                  volatility=0.02,
                  brownian_motion=brownian_motion)

starting_short_rate_value = 0.05
vm.generate_path(starting_state_space_values=starting_short_rate_value, set_path=True, seed=2024)

def test_conditional_mean() -> None:
    """ Tests that the conditional mean is within the allowed tolerance of the theoretical conditional mean. """
    PASS_THRESH = 1E-14

    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    admissible_dates = (date_obj for date_obj in dates if date_obj <= vm.end_date_time)

    for date_obj in admissible_dates:
        accrual = DayCountCalculator.compute_accrual_length(start_time, date_obj, vm.day_count_convention)
        weight = math.exp(-vm.reversion_speed * accrual)
        theoretical_val = starting_short_rate_value * weight + (1 - weight) * vm.long_term_mean
        result = abs(theoretical_val - vm.expected_short_rate(date_obj)) < PASS_THRESH
        assert result


def test_model_evaluates_to_path_on_interpolating_dates() -> None:
    """
    Tests the callable feature of the model correctly gives the path values
    when the datetime object used as an argument is an interpolation date.
    """
    PASS_THRESH = 1E-13
    date_range = pd.date_range(start=start_time, end=end_time, periods=len(vm.path)).to_pydatetime()

    for index, date_time_obj in enumerate(date_range):
        val = vm(date_time_obj)
        assert abs(val - vm.path[index]) < PASS_THRESH



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

    PASS_THRESH = 1E-13

    for accrual, date_obj in zip(accruals[1:], admissible_dates[1:]):
        vm._create_bond_price_coeffs(date_obj)
        vm._create_bond_yield_coeffs(date_obj)

        assert abs(vm.price_state_variable_coeffs['intercept']/accrual + vm.yield_state_variable_coeffs['intercept']) < PASS_THRESH
        assert abs(vm.price_state_variable_coeffs['coefficient']/accrual + vm.yield_state_variable_coeffs['coefficient']) < PASS_THRESH


def test_conditional_short_rate_plus_convexity_equals_yield() -> None:
    """
    This test ensures that the proper theoretical relationship between the conditional mean,
    the convexity adjustment, and the yield exists for the Vasicek model. Namely, we must have:
        Average conditional short rate + Convexity Adjustment = Time T yield.
    """
    PASS_THRESH = 1E-13
    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    admissible_dates = [date_obj for date_obj in dates if date_obj < vm.end_date_time]

    assert all(abs(vm.average_expected_short_rate(maturity_date=date_obj)
                   + vm.yield_convexity(maturity_date=date_obj)
                   - vm.zero_coupon_yield(maturity_date=date_obj)) < PASS_THRESH
               for date_obj in admissible_dates[1:])

def test_affine_coefficients_are_zero_at_maturity() -> None:
    """
    Tests that the affine model coefficients for the Vasicek Model are zero for the bond maturity date.
    """
    PASS_THRESH = 1E-13
    dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                                   end_date=end_time,
                                                   increment=timedelta(1),
                                                   max_dates=1_000_000)

    admissible_dates = (date_obj for date_obj in dates if date_obj <= vm.end_date_time)

    for date_obj in admissible_dates:
        vm._create_bond_price_coeffs(maturity_date=date_obj, purchase_date=date_obj)
        assert abs(vm.price_state_variable_coeffs['intercept']) < PASS_THRESH
        assert abs(vm.price_state_variable_coeffs['coefficient']) < PASS_THRESH


#----------------------------------------------------------------------
# Multivariate Vasicek Model

short_rate_intercept = 0.01
short_rate_coefficients = np.array([0.02, -0.001])
reversion_level = np.array([0.03, 0.05])

rho = 0.0
stand_dev_1 = 1
stand_dev_2 = 1
volatility_matrix = np.array([[stand_dev_1 ** 2, stand_dev_1 * stand_dev_2 * rho],
                               [stand_dev_1 * stand_dev_2 * rho, stand_dev_2 ** 2]])

brownian_motion = BrownianMotion(start_date_time=start_time,
                                 end_date_time=end_time,
                                 dimension=2)


reversion_directions = np.array([[1.0/math.sqrt(2), 1.0/math.sqrt(2)],
                                 [1.0/math.sqrt(2), -1.0/math.sqrt(2)]])
reversion_matrix = reversion_directions @ np.array([[0.02, 0.05], [0.05, 0.1]]) @ reversion_directions.T

mvm = MultivariateVasicekModel(
    short_rate_intercept=short_rate_intercept,
    short_rate_coefficients=short_rate_coefficients,
    reversion_level=reversion_level,
    reversion_matrix=reversion_matrix,
    volatility_matrix=volatility_matrix,
    brownian_motion=brownian_motion)

starting_state_variables = np.array([0.03, 0.02])
mvm.generate_path(starting_state_variables, set_path=True, seed=1)

dates = Scheduler.generate_dates_by_increments(start_date=start_time,
                                               end_date=end_time,
                                               increment=timedelta(1),
                                               max_dates=1_000_000)

admissible_dates = [date_obj for date_obj in dates if date_obj <= vm.end_date_time]

def test_multivariate_affine_coefficients_are_zero_at_maturity() -> None:
    """
    Tests that the affine model coefficients for the Multivariate Vasicek Model are zero for the bond maturity date.
    """
    PASS_THRESH = 1E-13

    for date_obj in admissible_dates:
        mvm._create_bond_price_coeffs(maturity_date=date_obj, purchase_date=date_obj)
        assert abs(mvm.price_state_variable_coeffs['intercept']) < PASS_THRESH
        assert np.sum(np.abs(mvm.price_state_variable_coeffs['coefficients'])) < PASS_THRESH

def test_multivariate_affine_yield_and_price_have_correct_relationship() -> None:
    """
    Tests that the multivariate Vasicek model yield (as calculated from using the affine coefficients) is
    always equal to -log(P_t^T)/(T-t) for the time-t bond price P_t^T of a zero-coupon bond maturing at time T.
    """
    PASS_THRESH = 1E-13

    for datetime_obj in admissible_dates[1:]:
        accrual = DayCountCalculator.compute_accrual_length(start_time, datetime_obj, mvm.day_count_convention)
        price = mvm.zero_coupon_bond_price(maturity_date=datetime_obj)
        bond_yield = mvm.zero_coupon_bond_yield(maturity_date=datetime_obj)
        transformed_yield = -math.log(price)/accrual
        assert abs(bond_yield - transformed_yield) < PASS_THRESH


def test_multivariate_affine_yield_raises_value_error_for_bad_maturity_date() -> None:
    """
    Tests that trying to calculate the yield of a zero-coupon bond whose maturity date is less-than-or-equal-to
    the purchase date results in a value error.
    """
    with pytest.raises(ValueError):
        mvm.zero_coupon_bond_yield(maturity_date=start_time)


def test_multivaraite_yield_volatility_raises_value_error_for_bad_maturity_date() -> None:
    """
    Tests that trying to calculate the yield volatility of a zero-coupon bond whose maturity date is
    less-than-or-equal-to the purchase date results in a value error being raises.
    """
    with pytest.raises(ValueError):
        mvm.yield_volatility(maturity_date=start_time)




#----------------------------------------------------------------------
# Testing results of when MultivariateVasicekModel is set to equal the standard Vasicek Model
PASS_THRESH = 1E-13

brownian_motion = BrownianMotion(start_date_time=start_time,
                                 end_date_time=end_time,
                                 dimension=1)

vm = VasicekModel(reversion_level=0.04,
                  reversion_speed=2.0,
                  volatility=0.02,
                  brownian_motion=brownian_motion)

starting_short_rate_value = 0.05
vm.generate_path(starting_state_space_values=starting_short_rate_value, set_path=True, seed=2024)

short_rate_intercept = 0.0
short_rate_coefficients = np.array([1.0])
reversion_level = np.array([0.04])
volatility_matrix = np.array([[0.02]])
reversion_matrix = np.array([[2.0]])

mvm = MultivariateVasicekModel(
    short_rate_intercept=short_rate_intercept,
    short_rate_coefficients=short_rate_coefficients,
    reversion_level=reversion_level,
    reversion_matrix=reversion_matrix,
    volatility_matrix=volatility_matrix,
    brownian_motion=brownian_motion)

starting_state_variables = np.array([0.05])
mvm.generate_path(starting_state_variables, set_path=True, seed=1)

def test_multivariate_affine_intercepts_are_same_as_vasicek_coefficients() -> None:
    """
    Tests that the multivariate Vasicek model has the same affine intercepts across a range of maturity dates
    as the standard Vasicek model when the multivariate parameters are set to replicate the single-variable model.
    """
    for datetime_obj in admissible_dates:
        mvm._create_bond_price_coeffs(maturity_date=datetime_obj)
        vm._create_bond_price_coeffs(maturity_date=datetime_obj)

        multi_intercept = mvm.price_state_variable_coeffs['intercept']
        vm_intercept = vm.price_state_variable_coeffs['intercept']

        assert abs(multi_intercept - vm_intercept) < PASS_THRESH


def test_multivariate_affine_coefficients_are_same_as_vasicek_coefficients() -> None:
    """
    Tests that the multivariate Vasicek model has the same affine coefficients across a range of maturity dates
    as the standard Vasicek model when the multivariate parameters are set to replicate the single-variable model.
    """

    for datetime_obj in admissible_dates:
        mvm._create_bond_price_coeffs(maturity_date=datetime_obj)
        vm._create_bond_price_coeffs(maturity_date=datetime_obj)
        multi_coeffs = mvm.price_state_variable_coeffs['coefficients']
        vm_coeffs = vm.price_state_variable_coeffs['coefficient']
        assert abs(multi_coeffs - vm_coeffs) < PASS_THRESH


def test_multivariate_yield_volatilities_are_same_as_vasicek_() -> None:
    """
    Tests that the multivariate Vasicek model has the same yield volatilities across a range of maturity dates
    as the standard Vasicek model when the multivariate parameters are set to replicate the single-variable model.
    """

    for datetime_obj in admissible_dates[1:]:
        vm_yield_vol = vm.yield_volatility(maturity_date=datetime_obj)
        multi_yield_vol = mvm.yield_volatility(maturity_date=datetime_obj)
        assert abs(multi_yield_vol - vm_yield_vol) < PASS_THRESH


def test_multivariate_coeff_deriv_wrt_maturity_are_same_as_vasicek() -> None:
    """
    Tests that the derivatives of the affine price coefficients for the multivariate Vasicek model
    and the Vasicek model are the same when the multivariate Vasicek model is made to be the same
    as the standard Vasicek model.
    """
    for datetime_obj in admissible_dates[1:]:
        vm_derivs = vm._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=datetime_obj)
        mvm_derivs = mvm._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=datetime_obj)

        assert abs(vm_derivs['coefficient'] - mvm_derivs['coefficients']) < PASS_THRESH


def test_multivariate_intercept_deriv_wrt_maturity_are_same_as_vasicek() -> None:
    """
    Tests that the derivatives of the affine price intercepts for the multivariate Vasicek model
    and the Vasicek model are the same when the multivariate Vasicek model is made to be the same
    as the standard Vasicek model.
    """
    for datetime_obj in admissible_dates[1:]:
        vm_derivs = vm._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=datetime_obj)
        mvm_derivs = mvm._calculate_bond_price_coeff_derivatives_wrt_maturity(maturity_date=datetime_obj)

        assert abs(vm_derivs['intercept'] - mvm_derivs['intercept']) < PASS_THRESH


def test_multivariate_instantaneous_forward_rates_are_same_as_vasicek() -> None:
    """
    Tests that the multivariate Vasicek model has the same instantaneous forward rates across a range of maturity dates
    as the standard Vasicek model when the multivariate parameters are set to replicate the single-variable model.
    """

    for datetime_obj in admissible_dates[1:]:
        vm_instantaneous_fr = vm.instantaneous_forward_rate(maturity_date=datetime_obj)
        multi_instantaneous_fr = mvm.instantaneous_forward_rate(maturity_date=datetime_obj)
        assert abs(vm_instantaneous_fr - multi_instantaneous_fr) < PASS_THRESH

def test_multivariate_expected_short_rates_are_same_as_vasicek() -> None:
    """
    Tests that the MiltivariateVasicekModel has the same expected short rate across a range of maturity dates
    as the standard Vasicek model when the multivariate parameters are set to replicate the single-variable model.
    """
    for datetime_obj in admissible_dates:
        vm_expected_sr = vm.expected_short_rate(maturity_date=datetime_obj)
        multi_expected_sr = mvm.expected_short_rate(maturity_date=datetime_obj)
        assert abs(vm_expected_sr - multi_expected_sr) < PASS_THRESH

def test_multivariate_short_rate_variances_are_same_as_vasicek() -> None:
    """
    Tests that the MiltivariateVasicekModel has the same expected short rate across a range of maturity dates
    as the standard Vasicek model when the multivariate parameters are set to replicate the single-variable model.
    """
    for datetime_obj in admissible_dates:
        vm_variance = vm.short_rate_variance(maturity_date=datetime_obj)
        multi_variance = mvm.short_rate_variance(maturity_date=datetime_obj)
        assert abs(vm_variance - multi_variance) < PASS_THRESH
