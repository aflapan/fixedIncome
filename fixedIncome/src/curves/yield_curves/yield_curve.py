"""
This script contains the implementation of two objects related to curve construction,
the YieldCurve object and the YieldCurveFactory object.
"""
from __future__ import annotations

import bisect
import os
import urllib.request
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import scipy  # type: ignore
from datetime import date
from enum import Enum
from typing import Callable, Optional, NamedTuple, Iterable
from functools import partial


from fixedIncome.src.curves.base_curve import Curve, DiscountCurve
from fixedIncome.src.curves.curve_enumerations import (InterpolationSpace,
                                                       InterpolationMethod,
                                                       CurveIndex,
                                                       EndBehavior,
                                                       KnotValuePair)

from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountCalculator, DayCountConvention
from fixedIncome.src.assets.base_cashflow import CashflowCollection
from fixedIncome.src.curves.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryInstrument, ONE_BASIS_POINT


class HedgeRatio(NamedTuple):
    dv01: float
    hedge_ratio: float
    key_rate_date: date


class YieldCurve(Curve):
    def __init__(self,
                 instruments: Iterable[UsTreasuryInstrument],
                 quote_adjustments: Optional[Iterable[KnotValuePair]],
                 interpolation_method: InterpolationMethod,
                 interpolation_day_count_convention: DayCountConvention,
                 interpolation_space: InterpolationSpace,
                 reference_date: date,
                 left_end_behavior: EndBehavior = EndBehavior.ERROR,
                 right_end_behavior: EndBehavior = EndBehavior.ERROR) -> None:

        instruments = list(instruments)
        quote_adjustments = list(quote_adjustments) if quote_adjustments is not None else []
        self._interpolation_values = [instrument.to_knot_value_pair() for instrument in instruments]

        #TODO: Consider logging the adjustments
        for kv_pair, quote_adjustment in zip(self.interpolation_values, quote_adjustments):
            kv_pair.value += quote_adjustment

        super().__init__(self.interpolation_values,
                         interpolation_method,
                         interpolation_day_count_convention,
                         reference_date,
                         left_end_behavior,
                         right_end_behavior)

        self.instruments = instruments
        self.quote_adjustments = quote_adjustments
        self.interpolation_space = interpolation_space
        self.discount_curve: Optional[DiscountCurve] = None


    # present value calculators
    def present_value(self, instrument: CashflowCollection,
                      adjustment_fxcn: Optional[Callable[[date], float]] = None) -> Optional[float]:
        """
        Wrapper function for the specific implementations of present value calculations.
        The curve first transforms into the discount curve, and then the instrument specific
        method is called on the instrument provided a discount curve.
        """
        discount_curve = self.to_discount_curve(adjustment_fxcn)
        return discount_curve.present_value(instrument)

    def to_discount_curve(self, adjustment_fxcn: Optional[Callable[[date], float]] = None) -> DiscountCurve:
        """
        Method to transform the YieldCurve into a DiscountCurve object depending
        on the interpolation space used in the Yield Curve.
        """

        match self.interpolation_space:
            case InterpolationSpace.DISCOUNT_FACTOR:

                interpolation_values = [KnotValuePair(knot=kv_pair.knot, value=self(kv_pair.knot, adjustment_fxcn))
                                        for kv_pair in self.interpolation_values]

                if self.discount_curve is None:

                    self.discount_curve = DiscountCurve(interpolation_values=interpolation_values,
                                                        interpolation_method=self.interpolation_method,
                                                        index=CurveIndex.US_TREASURY,
                                                        interpolation_day_count_convention=self.interpolation_day_count_convention,
                                                        reference_date=self.reference_date,
                                                        left_end_behavior=EndBehavior.ERROR,
                                                        right_end_behavior=EndBehavior.ERROR)
                else:
                    self.discount_curve.reset_interpolation_values(interpolation_values)

                return self.discount_curve

            case InterpolationSpace.CONTINUOUSLY_COMPOUNDED_YIELD:
                max_date = self.interpolation_values[-1].knot

                date_range = pd.date_range(start=self.reference_date,
                                           end=max_date,
                                           freq='D').date

                accruals = np.array([self.date_to_interpolation_axis(date_obj) for date_obj in date_range])
                yields = np.array([self(date_obj, adjustment_fxcn) for date_obj in date_range])
                discount_factors = np.exp(- yields * accruals)  # specific to continuously-compounded yields
                interpolation_values = [KnotValuePair(knot=date_obj, value=df)
                                        for date_obj, df in zip(date_range, discount_factors)]

                if self.discount_curve is None:
                    self.discount_curve = DiscountCurve(interpolation_values=interpolation_values,
                                                        interpolation_method=InterpolationMethod.LINEAR,  # because we fill the date range
                                                        index=CurveIndex.US_TREASURY,
                                                        interpolation_day_count_convention=self.interpolation_day_count_convention,
                                                        reference_date=self.reference_date,
                                                        left_end_behavior=EndBehavior.ERROR,
                                                        right_end_behavior=EndBehavior.ERROR)

                else:
                    self.discount_curve.reset_interpolation_values(interpolation_values)

                return self.discount_curve

            case _:
                return NotImplemented('Only DISCOUNT_FACTOR and CONTINUOUSLY_COMPOUNDED_YIELD '
                                      'are implemented in to_discount_curve.')


    #--------------------------------------------------------------
    # functionality for bumping curves

    def parallel_bump(self, bump_amount = ONE_BASIS_POINT) -> Callable[[date], float]:
        """
        Returns an adjustment function corresponding to a parallel shift (i.e. a constant of bump_amount).
        """
        return lambda date_obj: bump_amount

    #----------------------------------------------------------------------
    # Duration and convexity with respect to parallel shifts of yield curve
    def calculate_pv_deriv(self, bond, offset: float = 0.0) -> float:
        """
        Calculates the DV01 of the provided us_treasury_instruments under parallel shifts of the yield curve.
        offset allows the user to specify a shift around which the derivative will be computed.

        Formula is
            deriv = (PV(offset+half basis point) - PV(offset-half basis Point))/(0.005 - -0.005)
        """
        half_bp_adjustment = self.parallel_bump(bump_amount=offset + ONE_BASIS_POINT/2)
        pv_plus_half_bp = self.present_value(bond, adjustment_fxcn=half_bp_adjustment)
        negative_half_bp_adjustment = self.parallel_bump(bump_amount=offset - ONE_BASIS_POINT/2)
        pv_minus_half_bp = self.present_value(bond, adjustment_fxcn=negative_half_bp_adjustment)

        derivative = (pv_plus_half_bp - pv_minus_half_bp)/ONE_BASIS_POINT  # 0.01 = 0.005 - -0.005

        return derivative

    def duration(self, instrument: UsTreasuryInstrument) -> float:
        """
        Calculates the duration of the us_treasury_instruments, as
        -1/P * dP/dy where P is the us_treasury_instruments price.
        """
        derivative = self.calculate_pv_deriv(instrument)
        present_value = self.present_value(instrument)
        return -derivative/present_value

    def convexity(self, bond) -> float:
        """
        calculate the convexity of a us_treasury_instruments, defined as C := 1/P * d^2 P/d^2y
        Reference: Tuckman and Serrat, 4th ed. equation (4.14).
        """

        derivative_positive_bump = self.calculate_pv_deriv(bond, offset=ONE_BASIS_POINT/2)
        derivative_negative_bump = self.calculate_pv_deriv(bond, offset=-ONE_BASIS_POINT/2)
        second_derivative = (derivative_positive_bump - derivative_negative_bump) / ONE_BASIS_POINT
        present_value = self.present_value(bond)

        return second_derivative / present_value

    def dv01(self, bond, adjustment: KeyRate) -> float:
        """
        Calculates the DV01 with respect to a KeyRate.
        """

        adjustment.create_adjustment_function()  # creates the default adjustment function with 1 bp movement
        pv_with_adjustment = self.present_value(bond, adjustment)
        pv_without_adjustment = self.present_value(bond)
        key_rate_dv01 = -(pv_with_adjustment - pv_without_adjustment) / (ONE_BASIS_POINT * 100)  # 100 to convert into bps

        return key_rate_dv01

    def calculate_dv01s(self, bond, key_rate_collection: KeyRateCollection) -> list[HedgeRatio]:
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
        interpolation_timestamps = list(pd.date_range(
            start=self.reference_date, end=max([instrument.to_knot_value_pair().knot for
                                                instrument in self.instruments]),
            periods=200
        ).date)

        knot_points = [self.date_to_interpolation_axis(instrument.to_knot_value_pair().knot)
                       for instrument in self.instruments]
        knot_vals = [self(instrument.to_knot_value_pair().knot) * 100
                     for instrument in self.instruments]

        # inject knot points into interpolation dates
        for instrument in self.instruments:
            bisect.insort_right(interpolation_timestamps, instrument.to_knot_value_pair().knot)

        time = [self.date_to_interpolation_axis(date_obj) for date_obj in interpolation_timestamps]
        orig_y_vals = [self(date_obj) * 100 for date_obj in interpolation_timestamps]
        discount_curve = self.to_discount_curve()
        discount_vals = [discount_curve(date_obj) for date_obj in interpolation_timestamps]

        fig, ax1 = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(10)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.xlabel('Date - Log Scale')
        ax1.set_ylim((min(orig_y_vals)-0.5, max(orig_y_vals) + 0.25))
        plt.xscale("log")
        plt.ylabel('Yield (%)')
        plt.plot(time, orig_y_vals, color='forestgreen', label='Yield Curve')
        plt.plot(knot_points, knot_vals, color='forestgreen', marker='o', linestyle='')
        plt.xticks(knot_points, labels=[str(instrument.to_knot_value_pair().knot) for instrument in self.instruments])

        if adjustment is not None:
            adjusted_vals = [self(date_obj, adjustment) * 100 for date_obj in interpolation_timestamps]
            plt.plot(time, adjusted_vals, linestyle='dashed', color='forestgreen', label='Adjusted Yield Curve')
            discount_curve = self.to_discount_curve(adjustment_fxcn=adjustment)
            adjusted_discount_vals = [discount_curve(date_obj) for date_obj in interpolation_timestamps]


        ax2 = ax1.twinx()
        plt.ylabel('Discount Factor')
        ax2.set_ylim((0.0, 1.1))
        plt.plot(time, discount_vals, color='darkgrey', label='Discount Curve')
        if adjustment is not None:
            plt.plot(time, adjusted_discount_vals, linestyle='dashed', color='darkgrey', label='Adjusted Duscount Curve')

        title_string = f"{self.interpolation_space} Curve Interpolated " \
                       f"Using {self.interpolation_method.value.capitalize()} Method"

        plt.show()


    def plot_price_curve(self,
                         bond: UsTreasuryInstrument,
                         lower_shift: float = -200 * ONE_BASIS_POINT,
                         upper_shift: float = 200 * ONE_BASIS_POINT,
                         shift_increment: float = ONE_BASIS_POINT
                         ) -> None:
        """
        Plots the present value of the us_treasury_instruments along with linear (duration only) and
        quadratic (duration+convexity) approximations of the us_treasury_instruments price as the yield curve
        parallel shifts up and down.
        """

        deriv = self.calculate_pv_deriv(bond)
        bond_pv = self.present_value(bond)
        bond_convexty = self.convexity(bond)

        parallel_shifts = np.arange(start=lower_shift, stop=upper_shift+shift_increment, step=shift_increment)

        pv_vals = [self.present_value(bond, self.parallel_bump(shift)) for shift in parallel_shifts]
        linear_approx = [deriv * shift + bond_pv for shift in parallel_shifts]
        quad_approx = [bond_pv + deriv * shift + shift**2 * bond_pv*bond_convexty/2 for shift in parallel_shifts]

        plt.figure(figsize=(10, 6))
        plt.plot(parallel_shifts*10_000, pv_vals, color="black", linewidth=2)
        plt.plot(parallel_shifts*10_000, quad_approx, color="black", linestyle="-.", linewidth=1.5)
        plt.plot(parallel_shifts*10_000, linear_approx, color="black", linestyle=':', linewidth=1)
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

    def construct_curve_from_yield_to_maturities(self, instruments: Iterable[UsTreasuryInstrument],
                                                 interpolation_method: InterpolationMethod,
                                                 reference_date: date) -> YieldCurve:
        """
        Method to construct a YieldCurve object from the provided list of Bond objects and the
        interpolation method string.

        Returns a YieldCurve object calibrated to interpolate between
        """

        yield_curve = YieldCurve(
            instruments=instruments,
            quote_adjustments=None,
            interpolation_method=interpolation_method,
            interpolation_space=InterpolationSpace.YIELD_TO_MATURITY,
            interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
            reference_date=reference_date,
            left_end_behavior=EndBehavior.CONSTANT,
            right_end_behavior=EndBehavior.CONSTANT
        )

        return yield_curve

    def construct_yield_curve(self, instruments: Iterable[UsTreasuryInstrument],
                              interpolation_method: InterpolationMethod,
                              reference_date: date) -> YieldCurve:

        knot_dates = [bond_obj.maturity_date for bond_obj in instruments]
        values = [0.0 for date_obj in knot_dates]
        market_prices = np.array([bond_obj.price for bond_obj in instruments])

        # initialize the yield curve object
        yield_curve_obj = YieldCurve(
            instruments=instruments,
            quote_adjustments=None,
            interpolation_method=interpolation_method,
            interpolation_space=InterpolationSpace.CONTINUOUSLY_COMPOUNDED_YIELD,
            interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
            reference_date=reference_date,
            left_end_behavior=EndBehavior.CONSTANT,
            right_end_behavior=EndBehavior.CONSTANT
        )

        def value_to_pv_diffs(knot_values: np.array[float]) -> np.array:
            knot_values = [KnotValuePair(knot=instrument.to_knot_value_pair().knot, value=val)
                           for (instrument, val) in zip(instruments, knot_values)]
            yield_curve_obj.reset_interpolation_values(knot_values)
            pvs = np.array([yield_curve_obj.present_value(bond_obj) for bond_obj in instruments])  # markets quote clean
            return pvs - market_prices

        convergence_solution = scipy.optimize.root(value_to_pv_diffs,
                                                   x0=np.array([0 for _ in values]),
                                                   tol=1e-10)

        calibrated_values = convergence_solution['x']
        knot_values = [KnotValuePair(knot=instrument.to_knot_value_pair().knot, value=val)
                       for (instrument, val) in zip(instruments, calibrated_values)]
        yield_curve_obj.reset_interpolation_values(knot_values)

        return yield_curve_obj

    def bootstrap_yield_curve(self, instruments: Iterable[UsTreasuryInstrument],
                              interpolation_method: InterpolationMethod,
                              reference_date: date) -> YieldCurve:

        """
        Constructs a yield curve by bootstrapping the PV values to match their market prices.
        """
        instruments = list(instruments)

        market_prices = [bond_obj.price for bond_obj in instruments]

        # initialize the yield curve object
        yield_curve_obj = YieldCurve(
            instruments=instruments,
            quote_adjustments=None,
            interpolation_method=interpolation_method,
            interpolation_space=InterpolationSpace.CONTINUOUSLY_COMPOUNDED_YIELD,
            interpolation_day_count_convention=DayCountConvention.ACTUAL_OVER_ACTUAL,
            reference_date=reference_date,
            left_end_behavior=EndBehavior.CONSTANT,
            right_end_behavior=EndBehavior.CONSTANT
        )

        def value_to_pv_diff(knot_value: float, index: int,  target: float, instrument : UsTreasuryInstrument) -> float:
            """
            Helper function which computes the difference between the
            PV of an instrument computed on the yield curve and the target value.
            """
            knot_date = yield_curve_obj.interpolation_values[index].knot
            new_knot_value = KnotValuePair(knot=knot_date, value=float(knot_value))
            yield_curve_obj.reset_interpolation_value(new_value=new_knot_value, index=index)

            return yield_curve_obj.present_value(instrument) - target


        for index, instrument in enumerate(instruments):
            frzn_pv_diff = partial(value_to_pv_diff, index=index, target=market_prices[index], instrument=instrument)
            convergence_solution = scipy.optimize.root(frzn_pv_diff,
                                                       x0=np.array([0.0]),
                                                       tol=1e-10)

            solution_knot = KnotValuePair(knot=instrument.to_knot_value_pair().knot,
                                          value=float(convergence_solution['x']))

            if index == 4:
                yield_curve_obj.present_value(instrument)

        return yield_curve_obj

