"""
This script contains the implementation of two objects related to curve construction,
the YieldCurve object and the YieldCurveFactory object.
"""
from __future__ import annotations

import os
import urllib.request
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import scipy  # type: ignore
from datetime import date
from enum import Enum
from typing import Callable, Optional, NamedTuple, Sequence, Iterable
import functools

from fixedIncome.src.curves.base_curve import Curve, DiscountCurve, KnotValuePair, EndBehavior, InterpolationMethod, CurveIndex
from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator, DayCountConvention
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryBond, ONE_BASIS_POINT
from fixedIncome.src.curves.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.assets.base_cashflow import CashflowCollection

class HedgeRatio(NamedTuple):
    dv01: float
    hedge_ratio: float
    key_rate_date: date

class InterpolationSpace(Enum):
    DISCOUNT_FACTOR = 0
    FORWARD_RATES = 1
    YIELD = 2
    CONTINUOUSLY_COMPOUNDED_YIELD = 3
    YIELD_TO_MATURITY = 4
    CONTINUOUSLY_COMPOUNDED_YIELD_TO_MATURITY = 5


class YieldCurve(Curve):
    def __init__(self,
                 instruments: Iterable[CashflowCollection],
                 quote_adjustments: Optional[Iterable[KnotValuePair]],
                 interpolation_method: InterpolationMethod,
                 interpolation_day_count_convention: DayCountConvention,
                 interpolation_space: InterpolationSpace,
                 reference_date: date,
                 left_end_behavior: EndBehavior = EndBehavior.ERROR,
                 right_end_behavior: EndBehavior = EndBehavior.ERROR) -> None:

        instruments = list(instruments)
        quote_adjustments = list(quote_adjustments)
        interpolation_values = [instrument.to_knot_value_pair() for instrument in instruments]

        #TODO: Consider logging the adjustments
        for kv_pair, quote_adjustment in zip(interpolation_values, quote_adjustments):
            kv_pair.value += quote_adjustment

        super().__init__(interpolation_values,
                         interpolation_method,
                         interpolation_day_count_convention,
                         reference_date,
                         left_end_behavior,
                         right_end_behavior)

        self.instruments = instruments
        self.quote_adjustments = quote_adjustments
        self.interpolation_space = interpolation_space


    # present value calculators
    def present_value(self, instrument: UsTreasuryInstrument,
                      adjustment_fxcn: Optional[Callable[[date], float]] = None) -> Optional[float]:
        """
        Wrapper function for the specific implementations of present value calculations.
        The curve first transforms into the discount curve, and then the instrument specific
        method is called on the instrument provided a discount curve.
        """

        match self.interpolation_space:
            case InterpolationSpace.CONTINUOUSLY_COMPOUNDED_YIELD:
                pass

            case InterpolationSpace.DISCOUNT_FACTOR:
                pass

            case _:
                raise ValueError(f'Interpolation space {self.interpolation_space} does not')



    @functools.cached_property
    def to_discount_curve(self, adjustment_fxcn: Optional[Callable[[date], float]] = None) -> DiscountCurve:
        """ Method to transform the """

        match self.interpolation_space:
            case InterpolationSpace.DISCOUNT_FACTOR:
                return self

            case InterpolationSpace.CONTINUOUSLY_COMPOUNDED_YIELD:
                max_date = self.interpolation_values[-1].date

                date_range = pd.date_range(start=self.reference_date,
                                           end=max_date,
                                           freq='D').date

                accruals = np.array([self.date_to_interpolation_axis(date_obj) for date_obj in date_range])
                yields = np.array([self(date_obj, adjustment_fxcn) for date_obj in date_range])
                discount_factors = np.exp(- yields * accruals)  # specific to continuously-compounded yields
                interpolation_values = [KnotValuePair(knot=date_obj, value=df)
                                        for date_obj, df in zip(date_range, discount_factors)]

                return DiscountCurve(interpolation_values=interpolation_values,
                                     interpolation_method=InterpolationMethod.LINEAR,  # because we fill the date range
                                     index=CurveIndex.US_TREASURY,
                                     interpolation_day_count_convention=self.interpolation_day_count_convention,
                                     reference_date=self.reference_date,
                                     left_end_behavior=EndBehavior.ERROR,
                                     right_end_behavior=EndBehavior.ERROR)

            case _:
                return NotImplemented('Only DISCOUNT_FACTOR and CONTINUOUSLY_COMPOUNDED_YIELD '
                                      'are implemented in to_discount_curve.')





    #-------------------------------------------------------------------


    def _calc_pv_yield_space(self, bond: Bond, adjustment: Optional[Callable[[date], float]] = None) -> float:
        """
        Returns a float for the present value of a us_treasury_instruments.
        """

        received_payments = bond._is_payment_received(self.reference_date)

        time_to_payments = pd.Series(
            [
                DayCountCalculator.compute_accrual_length(
                    start_date=self.reference_date, end_date=adjusted_date, dcc=self.interpolation_dcc)
                for adjusted_date in bond.payment_schedule.loc[received_payments, 'Adjusted Date']
            ],
            index=bond.payment_schedule.loc[received_payments, 'Adjusted Date']
        )

        payment_amounts = bond.payment_schedule.loc[received_payments, ['Adjusted Date', 'Payment ($)']].set_index('Adjusted Date')
        payment_dates = bond.payment_schedule.loc[received_payments, 'Adjusted Date']

        yields = np.array([self.interpolate(pymnt_date, adjustment) for pymnt_date in payment_dates])

        # Divide yields by 100 to convert from % to absolute
        pv = np.exp(-(yields/100) * time_to_payments).dot(payment_amounts)  # sum_{i=1}^n Payment_i e^{-y_i * t_i}

        return pv.item()


    #--------------------------------------------------------------
    # functionality for bumping curves

    def parallel_bump(self, bump_amount = ONE_BASIS_POINT) -> Callable[[date], float]:
        """
        Returns an adjustment function corresponding to a parallel shift (i.e. a constant of bump_amount).
        """
        return lambda date_obj: bump_amount

    #----------------------------------------------------------------------
    # Duration and convexity with respect to parallel shifts of yield curve
    def calculate_pv_deriv(self, bond: Bond, offset: float = 0.0) -> float:
        """
        Calculates the DV01 of the provided us_treasury_instruments under parallel shifts of the yield curve.
        offset allows the user to specify a shift around which the derivative will be computed.


        Formula is
            deriv = (PV(offset+half basis point) - PV(offset-half basis Point))/(0.005 - -0.005)
        """
        half_bp_adjustment = self.parallel_bump(bump_amount=offset + 0.005)  # unit is in %, so 0.005 = half a basis point
        pv_plus_half_bp = self.calculate_present_value(bond, adjustment_fxcn=half_bp_adjustment)

        negative_half_bp_adjustment = self.parallel_bump(bump_amount=offset - 0.005)
        pv_minus_half_bp = self.calculate_present_value(bond, adjustment_fxcn=negative_half_bp_adjustment)

        derivative = (pv_plus_half_bp - pv_minus_half_bp)/0.01  # 0.01 = 0.005 - -0.005

        return derivative

    def duration(self, bond: Bond) -> float:
        """
        Calculates the duration of the us_treasury_instruments, as
        -1/P * dP/dy where P is the us_treasury_instruments price.
        """
        derivative = self.calculate_pv_deriv(bond)
        present_value = self.calculate_present_value(bond)
        return -derivative/present_value

    def convexity(self, bond: Bond) -> float:
        """
        calculate the convexity of a us_treasury_instruments, defined as C := 1/P * d^2 P/d^2y
        Reference: Tuckman and Serrat, 4th ed. equation (4.14).
        """

        derivative_positive_bump = self.calculate_pv_deriv(bond, offset=0.005)
        derivative_negative_bump = self.calculate_pv_deriv(bond, offset=-0.005)
        second_derivative = (derivative_positive_bump - derivative_negative_bump) / 0.01
        present_value = self.calculate_present_value(bond)

        return second_derivative / present_value

    def dv01(self, bond: Bond, adjustment: KeyRate) -> float:
        """
        Calculates the DV01 with respect to a KeyRate.
        """

        adjustment.create_adjustment_function()  # creates the default adjustment function with 1 bp movement

        pv_with_adjustment = self.calculate_present_value(bond, adjustment)

        pv_without_adjustment = self.calculate_present_value(bond)

        key_rate_dv01 = -(pv_with_adjustment - pv_without_adjustment) / (0.01 * 100)  # 100 to convert into bps

        return key_rate_dv01

    def calculate_dv01s(self, bond: Bond, key_rate_collection: KeyRateCollection) -> list[HedgeRatio]:
        """
        Computes the dv01s of the us_treasury_instruments with respect to each KeyRate in the KeyRateCollection.
        Returns a list of HedgeRatios.
        """

        dv01_list = [self.dv01(bond, key_rate) for key_rate in key_rate_collection]

        sum_of_dv01s = sum(dv01_list)

        hedge_ratios = [dv01/sum_of_dv01s for dv01 in dv01_list]

        key_rate_dates = [kr.key_rate_date for kr in key_rate_collection]

        return [HedgeRatio(dv01, hedge_ratio, key_rate_date) for
                (dv01, hedge_ratio, key_rate_date) in zip(dv01_list, hedge_ratios, key_rate_dates)]


    #----------------------------------------------------------------
    # plotting methods
    def plot(self, adjustment: Optional[Callable[[date], float]] = None) -> None:
        """
        Plots the yield curve object.
        """
        interpolation_timestamps = pd.date_range(
            start = self.reference_date, end=self.interpolation_values.index.max(), periods=200
        )

        orig_y_vals = [self.interpolate(date_obj) for date_obj in interpolation_timestamps.date]

        plot_series = pd.Series(orig_y_vals, index=interpolation_timestamps.date)
        title_string = f"{self.interpolation_space} Curve Interpolated " \
                       f"Using {self.interpolation_method.capitalize()} Method"

        plt.figure(figsize=(10, 6))
        plot_series.plot(title=title_string, ylabel=self.interpolation_space, color='black', linewidth=2)

        if adjustment is not None:
            bumped_y_vals = [self.interpolate(date_obj, adjustment) for date_obj in interpolation_timestamps.date]
            plot_bumped_series = pd.Series(bumped_y_vals, index=interpolation_timestamps.date)
            plot_bumped_series.plot(linestyle='--', color='black')

        plt.show()


    def plot_price_curve(self, bond: Bond,
                         lower_shift: float = -2.0, upper_shift: float = 2.0, shift_increment: float = 0.01) -> None:
        """
        Plots the present value of the us_treasury_instruments along with linear (duration only) and
        quadratic (duration+convexity) approximations of the us_treasury_instruments price as the yield curve
        parallel shifts up and down.
        """

        deriv = self.calculate_pv_deriv(bond)
        bond_pv = self.calculate_present_value(bond)
        bond_convexty = self.convexity(bond)

        parallel_shifts = np.arange(start=lower_shift, stop=upper_shift+shift_increment, step=shift_increment)

        pv_vals = [self.calculate_present_value(bond, self.parallel_bump(shift))
                   for shift in parallel_shifts]

        linear_approx = [deriv * shift + bond_pv for shift in parallel_shifts]
        quad_approx = [bond_pv + deriv * shift + shift**2 * bond_pv*bond_convexty/2 for shift in parallel_shifts]

        plt.figure(figsize=(10, 6))
        plt.plot(parallel_shifts*100, pv_vals, color="black", linewidth=2)
        plt.plot(parallel_shifts*100, quad_approx, color="black", linestyle="-.", linewidth=1.5)
        plt.plot(parallel_shifts*100, linear_approx, color="black", linestyle=':', linewidth=1)
        plt.xlabel("Shift in Basis Points (bp)")
        plt.ylabel(f"Present Value in USD ($)")
        plt.title(f'Present Value of {bond.tenor} Bond Across Parallel Shifts in the Yield Curve')
        plt.legend(['Present Value', 'Approximation with Duration and Convexity', 'Approximation with Duration'],
                   frameon=False)
        plt.grid(True, alpha=0.5, linewidth=0.5)
        plt.show()


class YieldCurveFactory(object):
    """
    A Factory object which creates YieldCurve objects.
    """

    #------------------------------------------------------------------
    # Methods for constructing a curve based on us_treasury_instruments Yields

    def construct_curve_from_yield_to_maturities(self, bond_collection: Sequence[Bond], interpolation_method: str,
                                                 reference_date: date) -> YieldCurve:
        """
        Method to construct a YieldCurve object from the provided list of Bond objects and the
        interpolation method string.

        Returns a YieldCurve object calibrated to interpolate between
        """

        maturity_yield_pairs = sorted([
            (bond.maturity_date, bond.calculate_yield_to_maturity(reference_date))
                                       for bond in bond_collection], key=lambda pair: pair[0])

        values = pd.Series([ytm for (maturity, ytm) in maturity_yield_pairs],
                           index=[maturity for (maturity, ytm) in maturity_yield_pairs])

        yield_curve = YieldCurve(
            interpolation_values=values,
            interpolation_method=interpolation_method,
            interpolation_space='Yield to Maturity',
            interpolation_day_count_convention='act/act',
            reference_date=reference_date,
            left_end_behavior='constant',
            right_end_behavior='constant'
        )

        return yield_curve

    def construct_yield_curve(self, bond_collection: Sequence[Bond], interpolation_method: str,
                              reference_date: date) -> YieldCurve:

        knot_dates = [bond_obj.maturity_date for bond_obj in bond_collection]

        values = pd.Series(0, index=knot_dates)

        market_prices = np.array([bond_obj.full_price for bond_obj in bond_collection])

        # initialize the yield curve object
        yield_curve_obj = YieldCurve(
            interpolation_values=values,
            interpolation_method=interpolation_method,
            interpolation_space='Yield',
            interpolation_day_count_convention='act/act',
            reference_date=reference_date,
            left_end_behavior='constant',
            right_end_behavior='constant'
        )

        def value_to_pv_diffs(knot_values: np.array) -> np.array:

            new_values = pd.Series(knot_values, index=values.index)

            yield_curve_obj.reset_interpolation_values(new_values)

            pvs = np.array([yield_curve_obj.calculate_present_value(bond_obj) for bond_obj in bond_collection])

            return pvs - market_prices

        convergence_solution = scipy.optimize.root(value_to_pv_diffs, x0=np.array([0 for _ in values]), tol=1e-10)

        calibrated_values = pd.Series(convergence_solution['x'], index=values.index)

        yield_curve_obj.reset_interpolation_values(calibrated_values)

        return yield_curve_obj


