"""
This script contains the implementation of two objects related to curve construction,
the YieldCurve object and the YieldCurveFactory object.
"""
from __future__ import annotations

import bisect
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import scipy  # type: ignore
from datetime import date
from typing import Callable, Optional, Iterable
from functools import partial


from fixedIncome.src.curves.base_curve import Curve, DiscountCurve, KnotValuePair, PresentValueable
from fixedIncome.src.curves.curve_enumerations import (InterpolationSpace,
                                                       InterpolationMethod,
                                                       CurveIndex,
                                                       EndBehavior)

from fixedIncome.src.scheduling_tools.day_count_calculator import DayCountConvention
from fixedIncome.src.risk.key_rate import KeyRate, KeyRateCollection
from fixedIncome.src.assets.us_treasury_instruments.us_treasury_instruments import UsTreasuryInstrument
from fixedIncome.src.risk.risk_metrics import ONE_BASIS_POINT, Risk, RiskLadder


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
    def present_value(self, instrument: PresentValueable,
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

    def calculate_key_rate_deriv(self, assets: PresentValueable, key_rate: KeyRate) -> float:
        """
        Calculates the hedge ratio of the assets with respect to a specified KeyRate,
        which is defined to be:
            (PV(plus key rate adjustment) - PV) / ONE BASIS POINT
        """
        key_rate.create_adjustment_function()  # creates the default adjustment function with 1 bp movement
        key_rate.set_bump_val(-ONE_BASIS_POINT/2)
        pv_with_negative_adjustment = self.present_value(assets, key_rate)

        key_rate.create_adjustment_function()
        key_rate.set_bump_val(ONE_BASIS_POINT / 2)
        pv_with_positive_adjustment = self.present_value(assets, key_rate)

        key_rate.create_adjustment_function()  # reset key rate bump value
        key_rate_deriv = (pv_with_positive_adjustment - pv_with_negative_adjustment) / ONE_BASIS_POINT
        return key_rate_deriv

    def calculate_key_rate_convexity(self,
                                     assets: PresentValueable,
                                     key_rate_collection: KeyRateCollection) -> list[float]:
        """
        """
        pass



    def calculate_pv01_risk(self,
                            assets: PresentValueable,
                            key_rate: KeyRate) -> Risk:
        """
        Calculates the risk of the assets, which is defined as the change in $ per 1 basis point
        movement with respect to the key rate adjustment function.
        """
        risk = self.calculate_key_rate_deriv(assets, key_rate) * ONE_BASIS_POINT
        return Risk(key_rate_date=key_rate.key_rate_date, pv01=risk, index=CurveIndex.US_TREASURY)

    def calculate_pv01_risk_ladder(self,
                                   assets: PresentValueable,
                                   key_rate_collection: KeyRateCollection) -> RiskLadder:
        """
        Returns the risk ladder of the assets on the yield curve.
        """
        pv01_risks = (self.calculate_pv01_risk(assets=assets, key_rate=key_rate) for key_rate in key_rate_collection)
        return RiskLadder(pv01_risks)




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

            if index == 4:  # TODO: What is this??
                yield_curve_obj.present_value(instrument)

        return yield_curve_obj

