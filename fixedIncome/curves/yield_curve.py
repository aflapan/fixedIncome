
import os
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import scipy  # type: ignore
from datetime import date
from typing import Callable, Optional, NamedTuple, Sequence


from fixedIncome.utils.day_count_calculator import DayCountCalculator
from fixedIncome.assets.bond import Bond



class KnotValuePair(NamedTuple):
    knot: float
    value: float

    def __str__(self):
        return f"knot point {self.knot}, value {self.value}"


class YieldCurve(object):

    def __init__(self,
                 interpolation_values: pd.Series,
                 interpolation_method: str,
                 interpolation_space: str,
                 interpolation_day_count_convention: str,
                 reference_date: date,
                 left_end_behavior: str,
                 right_end_behavior: str) -> None:
        """
        Creates an instance of a yield curve object.

        Parameters:
            interpolation_values: pd.Series whose indices are the dates used as knot points
            interpolation_method: A string representing the interpolation method to use. Valid inputs are
                                  'linear', ‘quadratic’, ‘cubic’, ‘previous’, 'forward'.
            interpolation_day_count_convention: A string representing the day count convention
            reference_date: The date from which
            left_end_behavior: A string determining the behavior of evaluating x values below the first
        """

        self.interpolation_values = interpolation_values
        self.interpolation_method = interpolation_method
        self.interpolation_space = interpolation_space
        self.interpolation_dcc = interpolation_day_count_convention
        self.reference_date = reference_date
        self.left_end_behavior = left_end_behavior
        self.right_end_behavior = right_end_behavior
        self.dcc_calculator_obj = DayCountCalculator()

        self.valid_interpolation_methods = {'linear', 'quadratic', 'cubic', 'previous'}

        self.valid_interpolation_spaces = {'Yield',
                                           'Yield to Maturity'
                                           }


        assert self.interpolation_method in self.valid_interpolation_methods
        assert self.interpolation_space in self.valid_interpolation_spaces

        self.knot_value_tuples_list = None
        self._preprocess_interpolation_values()
        self.interpolator = self._create_interpolation_object()


    #-----------------------------------------------------------
    # Functions for creating the interpolation method.

    def date_to_interpolation_axis(self, date: date) -> float:
        """
        Converts from a date to the interpolation x-axis value based on the
        reference date provided when the YieldCurve object is instantiated,
        the user-provided date, and the interpolation day-count convention string
        provided when the YieldCurve object is instantiated.
        day_count(reference_date, date) -> x-axis float value.
        """

        interpolation_axis_val = self.dcc_calculator_obj.compute_accrual_length(
            start_date=self.reference_date, end_date=date, dcc=self.interpolation_dcc
        )
        return interpolation_axis_val


    def _preprocess_interpolation_values(self) -> list[KnotValuePair]:
        """
        Pre-processes the interpolation dates into interpolation x-axis values.
        Returns an ordered list of namedTuples (x_val, y_val) of the x-axis knot points
        and the corresponding y values used for interpolation.

        Returns a list of KnotValuePair namedTuples, sorted by knot points.
        """

        self.interpolation_values.sort_index(inplace=True) # sorts by x values

        x_dates, y_values = self.interpolation_values.index.tolist(), self.interpolation_values.values.tolist()

        # preprocesses dates into interpolation x axis values
        x_values = [self.date_to_interpolation_axis(date_obj) for date_obj in x_dates]

        return [KnotValuePair(knot_val, y_val) for (knot_val, y_val) in zip(x_values, y_values)]


    def _create_interpolation_object(self) -> Callable[[float], float]:
        """
        Creates the interpolation object, a function which maps floats into floats.
        """

        self.knot_value_tuples_list = self._preprocess_interpolation_values()

        x_values = [knot_val_pair.knot for knot_val_pair in self.knot_value_tuples_list]
        y_values = [knot_val_pair.value for knot_val_pair in self.knot_value_tuples_list]

        interpolator = scipy.interpolate.interp1d(x=x_values,
                                                  y=y_values,
                                                  kind=self.interpolation_method,
                                                  assume_sorted=True)

        return interpolator


    def interpolate(self, date: date, adjustment: Optional[Callable[[date], float]] = None) -> float:
        """
        Method which uses the interpolation object
        """

        if adjustment is None:
            adjustment = lambda date_obj: 0

        x_axis_val = self.date_to_interpolation_axis(date)

        if x_axis_val < self.knot_value_tuples_list[0].knot:

            match self.left_end_behavior:
                case 'constant':
                    return self.knot_value_tuples_list[0].value + adjustment(date)
                case _:
                    raise ValueError(f'Error in interpolate. {self.left_end_behavior} not a valid input.')

        elif x_axis_val > self.knot_value_tuples_list[-1].knot:

            match self.right_end_behavior:
                case 'constant':
                    return self.knot_value_tuples_list[-1].value + adjustment(date)

                case _:
                    raise ValueError(f'Error in interpolate. {self.right_end_behavior} not a valid input.')

        else:
            return self.interpolator(x_axis_val) + adjustment(date)


    def reset_interpolation_values(self, new_values: pd.Series) -> None:
        """
        Resets the interpolator based on new (x,y) value pairs.
        """
        self.interpolation_values = new_values
        self.interpolator = self._create_interpolation_object()

    #-----------------------------------------------------------------
    # present value calculators
    def calculate_present_value(self, bond: Bond, adjustment_fxcn = None) -> Optional[float]:
        """
        Wrapper function for the specific implementations of present value calculations.
        """

        match self.interpolation_space:
            case 'Yield':
                return self._calc_pv_yield_space(bond, adjustment_fxcn)

            case 'Yield to Maturity':
                return self._calc_pv_yield_to_maturity_space(bond, adjustment_fxcn)

            case _:
                raise ValueError(f'PV calculation for interpolation space {self.interpolation_space} '
                                 f'not implemented.')

    def _calc_pv_yield_space(self, bond: Bond, adjustment: Optional[Callable[[date], float]] = None) -> float:
        """
        Returns a float for the present value of a bond.
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

    def parallel_bump(self, bump_amount = 0.01):
        """
        Returns an adjustment
        """
        return lambda date_obj: bump_amount

    #----------------------------------------------------------------------
    # Duration and convexity with respect to parallel shifts of yield curve
    def calculate_pv_deriv(self, bond: Bond, offset: float = 0.0) -> float:
        """
        Calculates the DV01 of the provided bond under parallel shifts of the yield curve.
        offset allows the user to specify a shift around which the derivative will be computed.


        Formula is
            deriv = (PV(offset+half basis point) - PV(offset-half basis Point))/(0.005 - -0.005)
        """
        half_bp_adjustment = self.parallel_bump(bump_amount=offset + 0.005)  # unit is in %, so 0.005 = half a basis point
        pv_plus_half_bp = self.calculate_present_value(bond, adjustment_fxcn=half_bp_adjustment)

        negative_half_bp_adjustment = self.parallel_bump(bump_amount=offset - 0.005)
        pv_minus_half_bp = self.calculate_present_value(bond, adjustment_fxcn=negative_half_bp_adjustment)

        deriv = (pv_plus_half_bp - pv_minus_half_bp)/0.01  # 0.01 = 0.005 - -0.005

        return deriv

    def duration(self, bond: Bond) -> float:
        """
        Calculates the duration of the bond, as
        -1/P * dP/dy where P is the bond price.
        """
        derivative = self.calculate_pv_deriv(bond)
        present_value = self.calculate_present_value(bond)  # should I calculate present value or can I just use price?
        return -derivative/present_value

    def convexity(self, bond: Bond) -> float:
        """
        calculate the convexity of a bond, defined as
        C: = 1/P * d^2 P/d^2y
        """

        derivative_positive_bump = self.calculate_pv_deriv(bond, offset=0.005)
        derivative_negative_bump = self.calculate_pv_deriv(bond, offset=-0.005)
        second_deriv = (derivative_positive_bump - derivative_negative_bump) / 0.01

        return second_deriv / self.calculate_present_value(bond)








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
        Plots the present value of the bond.
        """

        deriv = self.calculate_pv_deriv(bond)
        bond_pv = self.calculate_present_value(bond)
        bond_convexty = self.convexity(bond)

        parallel_shifts = np.arange(start=lower_shift, stop=upper_shift+shift_increment, step=shift_increment)

        pv_vals = [self.calculate_present_value(bond, self.parallel_bump(shift))
                   for shift in parallel_shifts]

        tangent_line = [deriv * shift + bond_pv for shift in parallel_shifts]

        quad_approx = [bond_pv + deriv * shift + shift**2 * bond_pv*bond_convexty/2 for shift in parallel_shifts]

        plt.figure(figsize=(10, 6))
        plt.plot(parallel_shifts*100, pv_vals, color="black", linewidth=2)
        plt.plot(parallel_shifts*100, quad_approx, color="black", linestyle="-.", linewidth=1.5)
        plt.plot(parallel_shifts*100, tangent_line, color="black", linestyle=':', linewidth=1)
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
    # Methods for constructing a curve based on bond Yields

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


    def read_data_from_auction_website(self, url_base="https://www.treasurydirect.gov/",
                                            url_extension="auctions/auction-query/") -> None:

        url = os.path.join(url_base, url_extension, "securities.csv")

        pass


